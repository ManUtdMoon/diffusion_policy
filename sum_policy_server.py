import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import click

from zprl.policy.zmq_policy_server import load_sum_policy, run_policy_server


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-b', '--base_ckpt', default=None, type=str)
@click.option('--bind_addr', default='tcp://127.0.0.1:5555')
def main(checkpoint, device, base_ckpt, bind_addr):
    policy = load_sum_policy(
        checkpoint=checkpoint,
        device=device,
        base_ckpt=base_ckpt)
    run_policy_server(policy, bind_addr)


if __name__ == '__main__':
    main()
