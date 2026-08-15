// tbv_ar.c — minimal 2-rank RDMA all-reduce transport for the Strix Halo
// USB4 link (usb4_rdma* verbs device from thunderbolt_ibverbs).
//
// Replaces RCCL's ~450us host-proxied all-reduce with a direct RDMA-write
// exchange at the ~65us wire floor: each rank writes its buffer into the
// peer's recv slot, spins on a trailing flag write (QP-ordered after data),
// and the caller (python) does the elementwise add on GPU via a zero-copy
// mapped view. SUM happens outside this lib.
//
// Build: gcc -O2 -shared -fPIC -o libtbv_ar.so tbv_ar.c -libverbs
//
// Protocol per call: data write -> flag write (same QP, ordered). Sequence
// number in the flag guards against reading a stale round.

#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>

#define TBV_MAX_BYTES (1u << 20)  /* per-slot staging; callers gate size */
#define TBV_NSLOTS 2              /* double-buffered recv: peer may run one
                                     round ahead without trampling the slot
                                     the local stream is still consuming */
#define TBV_GID_INDEX 1           /* RoCE v2 IPv4-mapped GID (perftest -x 1) */

static struct ibv_context *ctx;
static struct ibv_pd *pd;
static struct ibv_cq *cq;
static struct ibv_qp *qp;
static struct ibv_mr *mr_send, *mr_recv;
static struct ibv_mr *mr_flags;        /* host mode: aliases mr_recv */
static char *send_buf, *recv_buf;
static volatile uint64_t *recv_flag;   /* host mode: tail of recv_buf region;
                                          gpu mode: separate host flag page
                                          (the CPU spin can't deref GPU VAs) */
static uint64_t *send_flag;
static uint32_t peer_rkey_data, peer_rkey_flag;
static uint64_t peer_addr_data, peer_addr_flag;
static uint64_t seq;
static int inited;

char *tbv_send_ptr(void) { return send_buf; }
char *tbv_recv_ptr(void) { return recv_buf; }
uint32_t tbv_max_bytes(void) { return TBV_MAX_BYTES; }
uint32_t tbv_nslots(void) { return TBV_NSLOTS; }
uint64_t tbv_seq(void) { return seq; }

struct xchg { uint32_t qpn; uint32_t rkey_data; uint32_t rkey_flag;
              uint64_t addr_data; uint64_t addr_flag; uint8_t gid[16]; };

static int tcp_rendezvous(int rank, const char *peer_ip, int port,
                          struct xchg *mine, struct xchg *theirs) {
    int fd = -1;
    if (rank == 0) {
        int lfd = socket(AF_INET, SOCK_STREAM, 0);
        int one = 1;
        setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        struct sockaddr_in a = {0};
        a.sin_family = AF_INET; a.sin_port = htons(port);
        a.sin_addr.s_addr = INADDR_ANY;
        if (bind(lfd, (void*)&a, sizeof a) || listen(lfd, 1)) { perror("bind/listen"); return -1; }
        fd = accept(lfd, NULL, NULL);
        close(lfd);
    } else {
        for (int tries = 0; tries < 600; tries++) {
            fd = socket(AF_INET, SOCK_STREAM, 0);
            struct sockaddr_in a = {0};
            a.sin_family = AF_INET; a.sin_port = htons(port);
            inet_pton(AF_INET, peer_ip, &a.sin_addr);
            if (connect(fd, (void*)&a, sizeof a) == 0) break;
            close(fd); fd = -1;
            usleep(100000);
        }
        if (fd < 0) { fprintf(stderr, "tbv_ar: connect failed\n"); return -1; }
    }
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    if (write(fd, mine, sizeof *mine) != sizeof *mine) { perror("write"); return -1; }
    size_t got = 0;
    while (got < sizeof *theirs) {
        ssize_t r = read(fd, (char*)theirs + got, sizeof *theirs - got);
        if (r <= 0) { perror("read"); return -1; }
        got += r;
    }
    /* Keep the socket open: the caller runs a post-RTS barrier over it so neither
     * rank posts an RDMA WRITE before the peer's QP can receive it (a UC WRITE to
     * a not-yet-RTR QP is silently dropped -> permanent flag-spin deadlock). */
    return fd;
}

/* Post-RTS barrier over the rendezvous socket: both ranks write a byte then read
 * one, so each returns only once the peer has also reached this point (both QPs
 * in RTS).  Prevents the first-WRITE-before-peer-RTR drop race. */
static int tbv_barrier(int fd) {
    char c = 'R';
    if (write(fd, &c, 1) != 1) return -1;
    if (read(fd, &c, 1) != 1) return -1;
    return 0;
}

static int tbv_connect(int rank, const char *peer_ip, int port,
                       uint64_t my_addr_data, uint32_t my_rkey_data,
                       uint64_t my_addr_flag, uint32_t my_rkey_flag);

/* node_desc advertises "tbv peerN railM zc=1 ..." for zc-capable rails. */
static int tbv_dev_is_zc(const char *name) {
    char path[256], desc[128] = {0};
    snprintf(path, sizeof(path), "/sys/class/infiniband/%s/node_desc", name);
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    size_t n = fread(desc, 1, sizeof(desc) - 1, f);
    fclose(f);
    (void)n;
    return strstr(desc, "zc=1") != NULL;
}

static int tbv_open(void) {
    int num;
    struct ibv_device **devs = ibv_get_device_list(&num);
    if (!devs) { fprintf(stderr, "tbv_ar: no verbs devices\n"); return -1; }
    const char *want = getenv("DS4_TBV_AR_DEV"); /* explicit override */
    struct ibv_device *dev = NULL, *first = NULL, *zc = NULL;
    for (int i = 0; i < num; i++) {
        const char *name = ibv_get_device_name(devs[i]);
        if (want && *want) {
            if (!strcmp(name, want)) { dev = devs[i]; break; }
            continue;
        }
        if (strncmp(name, "usb4_rdma", 9))
            continue;
        if (!first) first = devs[i];
        if (!zc && tbv_dev_is_zc(name)) zc = devs[i];
    }
    /* prefer a rail with the zero-copy RX payload path; else first rail */
    if (!dev) dev = zc ? zc : first;
    if (!dev) { fprintf(stderr, "tbv_ar: no usb4_rdma device%s%s\n",
                        want ? " matching " : "", want ? want : "");
                ibv_free_device_list(devs); return -1; }
    fprintf(stderr, "tbv_ar: using %s%s\n", ibv_get_device_name(dev),
            (dev == zc) ? " (zc rail)" : "");
    ctx = ibv_open_device(dev);
    ibv_free_device_list(devs);
    if (!ctx) { fprintf(stderr, "tbv_ar: open_device failed\n"); return -1; }

    pd = ibv_alloc_pd(ctx);
    cq = ibv_create_cq(ctx, 64, NULL, NULL, 0);
    if (!pd || !cq) { fprintf(stderr, "tbv_ar: pd/cq failed\n"); return -1; }
    return 0;
}

int tbv_init(int rank, const char *peer_ip, int port) {
    if (inited) return 0;
    if (tbv_open()) return -1;

    /* staging: send | recv slots x2 | recv_flags(2x8B) | send_flag scratch */
    if (posix_memalign((void**)&send_buf, 4096, TBV_MAX_BYTES)) return -1;
    if (posix_memalign((void**)&recv_buf, 4096, TBV_NSLOTS * TBV_MAX_BYTES + 4096)) return -1;
    memset(send_buf, 0, TBV_MAX_BYTES);
    memset(recv_buf, 0, TBV_NSLOTS * TBV_MAX_BYTES + 4096);
    recv_flag = (volatile uint64_t *)(recv_buf + TBV_NSLOTS * TBV_MAX_BYTES);
    send_flag = (uint64_t *)(recv_buf + TBV_NSLOTS * TBV_MAX_BYTES + TBV_NSLOTS * 8);

    mr_send = ibv_reg_mr(pd, send_buf, TBV_MAX_BYTES,
                         IBV_ACCESS_LOCAL_WRITE);
    mr_recv = ibv_reg_mr(pd, recv_buf, TBV_NSLOTS * TBV_MAX_BYTES + 4096,
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mr_send || !mr_recv) { fprintf(stderr, "tbv_ar: reg_mr failed\n"); return -1; }
    mr_flags = mr_recv;   /* flags live in the recv_buf region's tail page */

    return tbv_connect(rank, peer_ip, port,
                       (uint64_t)recv_buf, mr_recv->rkey,
                       (uint64_t)recv_flag, mr_flags->rkey);
}

/*
 * GPU-direct init: the caller (python) owns the send/recv data buffers in
 * DEVICE memory (hipMalloc) and passes their dma-buf fds; data slots are
 * registered as dma-buf MRs so the RDMA WRITE lands straight in GPU pages.
 * Only the 8-byte round flags stay in a host page — the completion spin in
 * tbv_xfer runs on the CPU and can't dereference a GPU VA. Layout contract
 * matches tbv_init: send = TBV_MAX_BYTES, recv = TBV_NSLOTS * TBV_MAX_BYTES.
 */
int tbv_init_gpu(int rank, const char *peer_ip, int port,
                 uint64_t gpu_send_ptr, int send_fd, uint64_t send_off,
                 uint64_t gpu_recv_ptr, int recv_fd, uint64_t recv_off) {
    if (inited) return 0;
    if (tbv_open()) return -1;

    send_buf = (char *)gpu_send_ptr;
    recv_buf = (char *)gpu_recv_ptr;

    char *flag_page;
    if (posix_memalign((void**)&flag_page, 4096, 4096)) return -1;
    memset(flag_page, 0, 4096);
    recv_flag = (volatile uint64_t *)flag_page;
    send_flag = (uint64_t *)(flag_page + TBV_NSLOTS * 8);

    /* The dma-buf fd covers the WHOLE backing BO; the tensor may live at a
     * nonzero offset inside it (torch caching allocator suballocation).
     * Register with that offset or every RDMA lands offset-shifted in the
     * BO — silently corrupting NEIGHBORING allocations while the slots
     * read zeros. (Cost a fun evening.) */
    mr_send = ibv_reg_dmabuf_mr(pd, send_off, TBV_MAX_BYTES, gpu_send_ptr,
                                send_fd, IBV_ACCESS_LOCAL_WRITE);
    mr_recv = ibv_reg_dmabuf_mr(pd, recv_off, TBV_NSLOTS * TBV_MAX_BYTES,
                                gpu_recv_ptr, recv_fd,
                                IBV_ACCESS_LOCAL_WRITE |
                                IBV_ACCESS_REMOTE_WRITE);
    mr_flags = ibv_reg_mr(pd, flag_page, 4096,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mr_send || !mr_recv || !mr_flags) {
        fprintf(stderr, "tbv_ar: gpu reg_mr failed (send=%p recv=%p flags=%p)\n",
                (void*)mr_send, (void*)mr_recv, (void*)mr_flags);
        return -1;
    }

    return tbv_connect(rank, peer_ip, port,
                       gpu_recv_ptr, mr_recv->rkey,
                       (uint64_t)recv_flag, mr_flags->rkey);
}

static int tbv_connect(int rank, const char *peer_ip, int port,
                       uint64_t my_addr_data, uint32_t my_rkey_data,
                       uint64_t my_addr_flag, uint32_t my_rkey_flag) {
    struct ibv_qp_init_attr qia = {0};
    qia.send_cq = cq; qia.recv_cq = cq;
    /* UC: no retransmit engine -> the driver allows raw zero-copy TX
     * streams (RC is refused: a retransmitted raw stream could interleave
     * with a receiver mid-consumption). The TB link is lossless and the
     * data->flag ordering already carries correctness; RC ran 0 retransmits. */
    qia.qp_type = IBV_QPT_UC;
    qia.cap.max_send_wr = 32; qia.cap.max_recv_wr = 4;
    qia.cap.max_send_sge = 1; qia.cap.max_recv_sge = 1;
    qp = ibv_create_qp(pd, &qia);
    if (!qp) { fprintf(stderr, "tbv_ar: create_qp failed\n"); return -1; }

    union ibv_gid gid;
    if (ibv_query_gid(ctx, 1, TBV_GID_INDEX, &gid)) { fprintf(stderr, "tbv_ar: query_gid failed\n"); return -1; }

    struct xchg mine = { qp->qp_num, my_rkey_data, my_rkey_flag,
                         my_addr_data, my_addr_flag, {0} }, theirs;
    memcpy(mine.gid, gid.raw, 16);
    int rfd = tcp_rendezvous(rank, peer_ip, port, &mine, &theirs);
    if (rfd < 0) return -1;
    peer_rkey_data = theirs.rkey_data; peer_rkey_flag = theirs.rkey_flag;
    peer_addr_data = theirs.addr_data; peer_addr_flag = theirs.addr_flag;

    struct ibv_qp_attr at = {0};
    at.qp_state = IBV_QPS_INIT;
    at.pkey_index = 0; at.port_num = 1;
    at.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    if (ibv_modify_qp(qp, &at, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                      IBV_QP_ACCESS_FLAGS)) { fprintf(stderr, "tbv_ar: ->INIT failed\n"); return -1; }

    memset(&at, 0, sizeof at);
    at.qp_state = IBV_QPS_RTR;
    at.path_mtu = IBV_MTU_4096;
    at.dest_qp_num = theirs.qpn;
    at.rq_psn = 0;
    at.ah_attr.is_global = 1;
    memcpy(at.ah_attr.grh.dgid.raw, theirs.gid, 16);
    at.ah_attr.grh.sgid_index = TBV_GID_INDEX;
    at.ah_attr.grh.hop_limit = 64;
    at.ah_attr.port_num = 1;
    /* UC RTR/RTS: no RD-atomic, RNR, or retry attributes */
    if (ibv_modify_qp(qp, &at, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                      IBV_QP_DEST_QPN | IBV_QP_RQ_PSN)) {
        fprintf(stderr, "tbv_ar: ->RTR failed\n"); return -1;
    }

    memset(&at, 0, sizeof at);
    at.qp_state = IBV_QPS_RTS;
    at.sq_psn = 0;
    if (ibv_modify_qp(qp, &at, IBV_QP_STATE | IBV_QP_SQ_PSN)) {
        fprintf(stderr, "tbv_ar: ->RTS failed\n"); return -1;
    }

    /* Both QPs are now RTS.  Barrier so neither rank returns (and starts posting
     * WRITEs) until the peer is also RTS -- otherwise the first WRITE can hit a
     * not-yet-ready QP and be silently dropped on this UC link, deadlocking the
     * flag spin. */
    if (tbv_barrier(rfd)) { fprintf(stderr, "tbv_ar: post-RTS barrier failed\n"); close(rfd); return -1; }
    close(rfd);

    seq = 0;
    inited = 1;
    fprintf(stderr, "tbv_ar: rank%d ready (qpn=%u peer_qpn=%u)\n", rank, qp->qp_num, theirs.qpn);
    return 0;
}

/* Exchange nbytes: my send_buf -> peer recv_buf; returns when the peer's
 * round-`seq` data has fully landed in my recv_buf. */
int tbv_xfer(uint32_t nbytes) {
    if (!inited || nbytes > TBV_MAX_BYTES) return -1;
    seq++;
    uint32_t slot = (uint32_t)(seq % TBV_NSLOTS);

    struct ibv_sge sge_d = { (uint64_t)send_buf, nbytes, mr_send->lkey };
    struct ibv_send_wr wr_d = {0}, *bad;
    wr_d.wr_id = 1;
    wr_d.sg_list = &sge_d; wr_d.num_sge = 1;
    wr_d.opcode = IBV_WR_RDMA_WRITE;
    wr_d.wr.rdma.remote_addr = peer_addr_data + (uint64_t)slot * TBV_MAX_BYTES;
    wr_d.wr.rdma.rkey = peer_rkey_data;

    *send_flag = seq;
    struct ibv_sge sge_f = { (uint64_t)send_flag, 8, mr_flags->lkey };
    struct ibv_send_wr wr_f = {0};
    wr_f.wr_id = 2;
    wr_f.sg_list = &sge_f; wr_f.num_sge = 1;
    wr_f.opcode = IBV_WR_RDMA_WRITE;
    wr_f.send_flags = IBV_SEND_SIGNALED;   /* one completion per round */
    wr_f.wr.rdma.remote_addr = peer_addr_flag + (uint64_t)slot * 8;
    wr_f.wr.rdma.rkey = peer_rkey_flag;
    wr_d.next = &wr_f;

    if (ibv_post_send(qp, &wr_d, &bad)) { fprintf(stderr, "tbv_ar: post_send failed\n"); return -1; }

    /* wait for my sends to complete and the peer's round to arrive */
    struct ibv_wc wc;
    int got_cqe = 0;
    volatile uint64_t *rf = recv_flag + slot;
    /* Bounded spin: a genuine transfer lands in well under a millisecond, so a
     * multi-second wait means the peer WRITE never arrived. Return -1 instead of
     * deadlocking so the caller can latch off and fall back to the host reduce. */
    uint64_t spins = 0;
    const uint64_t SPIN_LIMIT = 20000000000ull;  /* ~ many seconds of busy poll */
    while (!got_cqe || *rf < seq) {
        if (!got_cqe) {
            int n = ibv_poll_cq(cq, 1, &wc);
            if (n < 0) { fprintf(stderr, "tbv_ar: poll_cq failed\n"); return -1; }
            if (n == 1) {
                if (wc.status != IBV_WC_SUCCESS) {
                    fprintf(stderr, "tbv_ar: wc status %d\n", wc.status);
                    return -1;
                }
                got_cqe = 1;
            }
        }
        if (++spins > SPIN_LIMIT) {
            fprintf(stderr, "tbv_ar: xfer TIMEOUT seq=%llu slot=%u got_cqe=%d recv_flag[slot]=%llu "
                            "(peer WRITE never landed) nbytes=%u\n",
                    (unsigned long long)seq, slot, got_cqe, (unsigned long long)*rf, nbytes);
            return -1;
        }
    }
    return 0;
}

/* --- stream host-callback entry point ------------------------------------
 * hipLaunchHostFunc-compatible shim: runs on HIP's callback thread in stream
 * order (after the D2H staging copy, before the H2D recv copy). Pure C --
 * no GIL, no HIP calls. Errors latch into tbv_err for the next python check. */
static volatile int tbv_err;
int tbv_last_error(void) { int e = tbv_err; tbv_err = 0; return e; }
void tbv_xfer_hostfn(void *arg) {
    if (tbv_xfer((uint32_t)(uintptr_t)arg) != 0)
        tbv_err = 1;
}
