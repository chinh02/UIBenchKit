#!/usr/bin/env python3
"""
Run lifecycle task helpers.
"""

import datetime
import os
import shutil


def create_run_experiment_task(
    *,
    ensure_dir,
    get_image_files,
    get_bot,
    generate_dcgen,
    generate_latcoder,
    generate_uicopilot,
    generate_layoutcoder,
    generate_single,
    prompt_direct: str,
    seg_params_default: dict,
    take_screenshots_task,
    run_evaluation_for_run,
):
    """Create a background run execution function bound to service dependencies."""
    def run_experiment_task(run):
        try:
            if run.status == "stopped":
                run.completed_at = datetime.datetime.now().isoformat()
                run.save_to_disk()
                return

            run.status = "running"
            ensure_dir(run.output_dir)

            filelist = get_image_files(run.input_dir)
            run.total_instances = len(filelist)

            for img_path in filelist:
                instance_id = os.path.basename(img_path).replace(".png", "")
                run.instances[instance_id] = {"status": "pending", "result": None}

            run.save_to_disk()

            if run.status == "stopped":
                run.completed_at = datetime.datetime.now().isoformat()
                run.save_to_disk()
                return

            bot = get_bot(run.model, user_api_key=run.user_api_key, user_base_url=run.user_base_url)
            if hasattr(bot, "reset_token_usage"):
                bot.reset_token_usage()

            save_interval = max(1, len(filelist) // 20)
            for idx, img_path in enumerate(filelist):
                if run.status == "stopped":
                    break

                instance_id = os.path.basename(img_path).replace(".png", "")
                filename = f"{instance_id}.html"
                save_path = os.path.join(run.output_dir, filename)
                run.instances[instance_id]["status"] = "running"

                try:
                    if run.method == "dcgen":
                        generate_dcgen(bot, img_path, save_path, seg_params_default)
                    elif run.method == "latcoder":
                        generate_latcoder(bot, img_path, save_path)
                    elif run.method == "uicopilot":
                        generate_uicopilot(bot, img_path, save_path)
                    elif run.method == "layoutcoder":
                        generate_layoutcoder(bot, img_path, save_path)
                    else:
                        generate_single(prompt_direct, bot, img_path, save_path)

                    run.instances[instance_id]["status"] = "completed"
                    run.instances[instance_id]["result"] = save_path
                except Exception as error:
                    run.instances[instance_id]["status"] = "failed"
                    run.instances[instance_id]["error"] = str(error)

                if (idx + 1) % save_interval == 0:
                    if hasattr(bot, "print_token_usage"):
                        run.token_usage = bot.print_token_usage(run.method)
                    run.save_to_disk()

                if run.status == "stopped":
                    break

            if run.status == "stopped":
                if hasattr(bot, "print_token_usage"):
                    run.token_usage = bot.print_token_usage(run.method)
                run.completed_at = datetime.datetime.now().isoformat()
                run.save_to_disk()
                return

            placeholder_src = os.path.join(run.input_dir, "placeholder.png")
            if os.path.exists(placeholder_src):
                shutil.copy(placeholder_src, os.path.join(run.output_dir, "placeholder.png"))

            take_screenshots_task(run.output_dir, replace=True)

            if hasattr(bot, "print_token_usage"):
                run.token_usage = bot.print_token_usage(run.method)

            run.evaluation = run_evaluation_for_run(run)
            run.status = "completed"
            run.completed_at = datetime.datetime.now().isoformat()
            run.save_to_disk()
        except Exception as error:
            run.status = "failed"
            run.error = str(error)
            run.completed_at = datetime.datetime.now().isoformat()
            run.save_to_disk()

    return run_experiment_task


def load_existing_runs(*, results_dir: str, runs_db: dict, run_cls) -> int:
    """Load persisted runs from disk into in-memory run database."""
    if not os.path.exists(results_dir):
        return 0

    loaded = 0
    for run_dir_name in os.listdir(results_dir):
        run_dir = os.path.join(results_dir, run_dir_name)
        if not os.path.isdir(run_dir):
            continue
        if run_dir_name in runs_db:
            continue

        run = run_cls.load_from_disk(run_dir)
        if run:
            runs_db[run.run_id] = run
            loaded += 1

    return loaded
