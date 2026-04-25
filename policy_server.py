import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import click

from zprl.policy.zmq_policy_server import load_base_policy, run_policy_server


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-t', '--n_action_steps', default=4, type=int, required=True)
@click.option('-n', '--num_inference_steps', default=5, type=int, required=False)
@click.option('--bind_addr', default='tcp://127.0.0.1:5555')
def main(checkpoint, device, n_action_steps, num_inference_steps, bind_addr):
    policy = load_base_policy(
        checkpoint=checkpoint,
        device=device,
        n_action_steps=n_action_steps,
        num_inference_steps=num_inference_steps)
    run_policy_server(policy, bind_addr)


if __name__ == '__main__':
    main()
