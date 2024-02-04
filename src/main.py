import argparse
import sys

import tug_command
import sts_command

parser = argparse.ArgumentParser(prog="posalyzer")
subparsers = parser.add_subparsers(dest="command", metavar="command")

parser_tug = subparsers.add_parser("tug")
tug_command.configure(parser_tug)

parser_sts = subparsers.add_parser("sts")
sts_command.configure(parser_sts)

print("-----------------------------")
print("    Nightingale Posalyzer    ")
print("-----------------------------")


args = parser.parse_args()

def print_help():
  print("Available commands:\n")
  print("\tposalyzer tug")
  print("\tposalyzer sts")
  print("\tposalyzer help")
  print()
  print("For help on a specific command, type: posalyzer COMMAND --help")
  sys.exit()

if args.command == "tug":
  tug_command.run(args)
elif args.command == "sts":
  sts_command.run(args)
elif args.command is None:
  parser.print_help()
  sys.exit(1)
