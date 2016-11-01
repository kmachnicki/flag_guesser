#!/usr/bin/env python3

from dataset import DataSet


def main():
    ds = DataSet()
    with open("data_sets/flag.csv",
              "r", newline='', encoding="utf8") as csv_file:
        ds.extract_from_csv(csv_file)

if __name__ == "__main__":
    main()