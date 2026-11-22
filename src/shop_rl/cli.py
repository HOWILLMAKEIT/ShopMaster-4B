"""shop_rl 统一命令入口。

用法：shop_rl <组> <命令> [参数…]；等价 python -m shop_rl.<组>.<模块>。
路由只看前两个词，其余参数原样交给模块自带的 argparse。
"""
import runpy
import sys

COMMANDS = {
    "data": {
        "export": "shop_rl.data.export_sft",
        "bands": "shop_rl.data.length_bands",
        "align": "shop_rl.data.align_task_ids",
    },
    "train": {
        "run": "shop_rl.train.sft",          # torchrun 用法见 AGENTS.md
        "merge": "shop_rl.train.merge_lora",
        "graft": "shop_rl.train.graft",
    },
    "eval": {
        "run": "shop_rl.eval.run",
        "summarize": "shop_rl.eval.summarize",
    },
    "rl": {
        "prepare-tasks": "shop_rl.rl.prepare_tasks",
        "monitor": "shop_rl.rl.monitor",
    },
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        for g, cmds in sorted(COMMANDS.items()):
            print(f"  shop_rl {g} <{'|'.join(sorted(cmds))}>")
        return 0
    group, cmd = argv[0], argv[1] if len(argv) > 1 else ""
    cmds = COMMANDS.get(group)
    if cmds is None:
        print(f"未知组：{group}", file=sys.stderr)
        return 2
    if cmd not in cmds:
        print(f"用法：shop_rl {group} <{'|'.join(sorted(cmds))}>", file=sys.stderr)
        return 2
    mod = cmds[cmd]
    sys.argv = [mod] + argv[2:]
    runpy.run_module(mod, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
