import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import click

from diffusion_policy.policy.zmq_policy_server import load_sum_policy, run_policy_server


@click.command()
@click.option("-c", "--checkpoint", required=True, help="Online residual checkpoint path.")
@click.option("-d", "--device", default="cuda:0")
@click.option("-t", "--n_action_steps", default=24, type=int, required=True)
@click.option("-s", "--num_inference_steps", default=10, type=int, show_default=True)
@click.option("-b", "--base_ckpt", default=None, type=str)
@click.option("--bind_addr", default="tcp://127.0.0.1:5555", show_default=True)
def main(checkpoint, device, n_action_steps, num_inference_steps, base_ckpt, bind_addr):
    policy = load_sum_policy(
        checkpoint=checkpoint,
        device=device,
        n_action_steps=n_action_steps,
        num_inference_steps=num_inference_steps,
        base_ckpt=base_ckpt,
    )
    run_policy_server(policy, bind_addr)


if __name__ == "__main__":
    main()
