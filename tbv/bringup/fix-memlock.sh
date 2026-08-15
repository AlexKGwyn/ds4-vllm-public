#!/usr/bin/env bash
# fix-memlock.sh — raise RDMA memory-lock limit (needed for RCCL/ibv_reg_mr).
# Run on BOTH boxes:  sudo ~/tbv/fix-memlock.sh
# No reboot needed — a NEW login session (or new ssh) picks it up via PAM.
set -e
cat > /etc/security/limits.d/99-rdma-memlock.conf <<'EOF'
*    soft    memlock    unlimited
*    hard    memlock    unlimited
root soft    memlock    unlimited
root hard    memlock    unlimited
EOF
# also raise the systemd (user+system) default so podman/toolbox inherit it
mkdir -p /etc/systemd/system.conf.d /etc/systemd/user.conf.d
printf '[Manager]\nDefaultLimitMEMLOCK=infinity\n' | tee /etc/systemd/system.conf.d/90-memlock.conf /etc/systemd/user.conf.d/90-memlock.conf >/dev/null
systemctl daemon-reexec 2>/dev/null || true
echo "memlock limit raised. Open a NEW shell/ssh session and verify: ulimit -l  (should say 'unlimited')"
