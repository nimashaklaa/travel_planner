import os
import sys

from pandas import DataFrame
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Cities:
    def __init__(self ,path=None) -> None:
        if path is None:
            # 1. Get the folder containing apis.py
            this_dir = os.path.dirname(os.path.abspath(__file__))
            # 2. Build the path to the CSV relative to apis.py
            path = os.path.join(
                this_dir,
                "..", "..",  # up two levels: from tools/flights/ to planner_agent/
                "database",
                "background",
                "citySet_with_states.txt"
            )
        self.path = path
        self.load_data()
        print("Cities loaded.")

    def load_data(self):
        cityStateMapping = open(self.path, "r").read().strip().split("\n")
        self.data = {}
        for unit in cityStateMapping:
            city, state = unit.split("\t")
            if state not in self.data:
                self.data[state] = [city]
            else:
                self.data[state].append(city)
    
    def run(self, state) -> dict:
        if state not in self.data:
            return ValueError("Invalid State")
        else:
            return self.data[state]