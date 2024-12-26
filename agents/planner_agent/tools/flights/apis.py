import os
import sys

import pandas as pd
from pandas import DataFrame
from typing import Optional
from agents.planner_agent.utils.func import extract_before_parenthesis
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "database/flights")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Flights:

    def __init__(self, path=None):
        if path is None:
            # 1. Get the folder containing apis.py
            this_dir = os.path.dirname(os.path.abspath(__file__))
            # 2. Build the path to the CSV relative to apis.py
            path = os.path.join(
                this_dir,
                "..", "..",  # up two levels: from tools/flights/ to planner_agent/
                "database",
                "flights",
                "clean_Flights_2022.csv"
            )

        self.path = path
        self.path = path
        self.data = None

        self.data = pd.read_csv(self.path).dropna()[['Flight Number', 'Price', 'DepTime', 'ArrTime', 'ActualElapsedTime','FlightDate','OriginCityName','DestCityName','Distance']]
        print("Flights API loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path).dropna().rename(columns={'Unnamed: 0': 'Flight Number'})

    def run(self,
            origin: str,
            destination: str,
            departure_date: str,
            ) -> DataFrame:
        """Search for flights by origin, destination, and departure date."""
        results = self.data[self.data["OriginCityName"] == origin]
        results = results[results["DestCityName"] == destination]
        results = results[results["FlightDate"] == departure_date]
        # if order == "ascPrice":
        #     results = results.sort_values(by=["Price"], ascending=True)
        # elif order == "descPrice":
        #     results = results.sort_values(by=["Price"], ascending=False)
        # elif order == "ascDepTime":
        #     results = results.sort_values(by=["DepTime"], ascending=True)
        # elif order == "descDepTime":
        #     results = results.sort_values(by=["DepTime"], ascending=False)
        # elif order == "ascArrTime":
        #     results = results.sort_values(by=["ArrTime"], ascending=True)
        # elif order == "descArrTime":
        #     results = results.sort_values(by=["ArrTime"], ascending=False)
        if len(results) == 0:
            return "There is no flight from {} to {} on {}.".format(origin, destination, departure_date)
        return results
    
    def run_for_annotation(self,
            origin: str,
            destination: str,
            departure_date: str,
            ) -> DataFrame:
        """Search for flights by origin, destination, and departure date."""
        results = self.data[self.data["OriginCityName"] == extract_before_parenthesis(origin)]
        results = results[results["DestCityName"] == extract_before_parenthesis(destination)]
        results = results[results["FlightDate"] == departure_date]
        # if order == "ascPrice":
        #     results = results.sort_values(by=["Price"], ascending=True)
        # elif order == "descPrice":
        #     results = results.sort_values(by=["Price"], ascending=False)
        # elif order == "ascDepTime":
        #     results = results.sort_values(by=["DepTime"], ascending=True)
        # elif order == "descDepTime":
        #     results = results.sort_values(by=["DepTime"], ascending=False)
        # elif order == "ascArrTime":
        #     results = results.sort_values(by=["ArrTime"], ascending=True)
        # elif order == "descArrTime":
        #     results = results.sort_values(by=["ArrTime"], ascending=False)
        return results.to_string(index=False)

    def get_city_set(self):
        city_set = set()
        for unit in self.data['data']:
            city_set.add(unit[5])
            city_set.add(unit[6])