// SPDX-License-Identifier: GPL-2.0
/*
 * nhi_throttle — set USB4/Thunderbolt NHI MSI-X interrupt throttling live.
 *
 * The mainline thunderbolt driver hardcodes ~128us interrupt moderation on
 * every NHI vector (REG_INT_THROTTLING_RATE 0x38c00 + 4*vector, 256ns units).
 * That is the measured ~65us mean one-way latency floor of usb4_rdma.
 * This module rewrites the registers on every ACTIVE NHI at load time and
 * stays loaded (rmmod + insmod with a new value to change it again).
 *
 *   insmod nhi_throttle.ko ns=8000
 */
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/pm_runtime.h>

static uint ns = 8000;
module_param(ns, uint, 0444);
MODULE_PARM_DESC(ns, "interrupt throttling interval in ns (256ns granularity)");

#define NHI_CLASS       0x0c0340
#define REG_INT_THROTTLE 0x38c00
#define NVEC            16

static int __init nhit_init(void)
{
	struct pci_dev *pdev = NULL;
	u32 val = min_t(u32, DIV_ROUND_UP(ns, 256), 0xffff);
	int hit = 0;

	while ((pdev = pci_get_class(NHI_CLASS, pdev))) {
		void __iomem *base;
		int i;

		if (pm_runtime_status_suspended(&pdev->dev)) {
			pr_info("nhi_throttle: %s suspended, skipping\n",
				pci_name(pdev));
			continue;
		}
		base = pci_iomap(pdev, 0, 0);
		if (!base) {
			pr_warn("nhi_throttle: %s iomap failed\n",
				pci_name(pdev));
			continue;
		}
		for (i = 0; i < NVEC; i++) {
			u32 off = REG_INT_THROTTLE + 4 * i;
			u32 old = ioread32(base + off);

			iowrite32(val, base + off);
			if (i < 2 || old != val)
				pr_info("nhi_throttle: %s vec%d %u -> %u (x256ns)\n",
					pci_name(pdev), i, old,
					ioread32(base + off));
		}
		pci_iounmap(pdev, base);
		hit++;
	}
	pr_info("nhi_throttle: set %u ns (%u x256ns) on %d NHI(s)\n",
		ns, val, hit);
	return hit ? 0 : -ENODEV;
}

static void __exit nhit_exit(void) {}

module_init(nhit_init);
module_exit(nhit_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Set USB4 NHI interrupt throttling rate");
