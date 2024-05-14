#!/usr/bin/env python
import argparse
from tira.local_execution_integration import LocalExecutionIntegration

def parse_args():
    parser = argparse.ArgumentParser(prog = 'tira-run')
    parser.add_argument('--input-directory', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--command', required=True)
    parser.add_argument('--output-directory', required=True)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    tira_execution = LocalExecutionIntegration(None)
    tira_execution.run(identifier=None, image=args.image, command=args.command, input_dir=args.input_directory, output_dir=args.output_directory)


