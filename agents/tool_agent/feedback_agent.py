import re, string, os, sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../planner_agent")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "tools/planner")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../tools/planner")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import importlib
from typing import List, Dict, Any
import tiktoken
from pandas import DataFrame
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from agents. tool_agent. prompts import  zeroshot_feedback_agent_prompt
import sys
import openai
import time
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
import os

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']

pd.options.display.max_info_columns = 200

os.environ['TIKTOKEN_CACHE_DIR'] = './tmper'

actionMapping = {"FlightSearch": "flights", "AttractionSearch": "attractions",
                 "GoogleDistanceMatrix": "googleDistanceMatrix", "AccommodationSearch": "accommodation",
                 "RestaurantSearch": "restaurants", "Planner": "planner", "NotebookWrite": "notebook",
                 "CitySearch": "cities"}


class CityError(Exception):
    pass


class DateError(Exception):
    pass


def catch_openai_api_error():
    error = sys.exc_info()[0]
    if error == openai.error.APIConnectionError:
        print("APIConnectionError")
    elif error == openai.error.RateLimitError:
        print("RateLimitError")
        time.sleep(60)
    elif error == openai.error.APIError:
        print("APIError")
    elif error == openai.error.AuthenticationError:
        print("AuthenticationError")
    else:
        print("API error:", error)


def load_tools(tools: List[str], planner_model_name=None) -> Dict[str, Any]:
    tools_map = {}
    for tool_name in tools:
        module = importlib.import_module("tools.{}.apis".format(tool_name))

        # Avoid instantiating the planner tool twice
        if tool_name == 'planner' and planner_model_name is not None:
            tools_map[tool_name] = getattr(module, tool_name[0].upper() + tool_name[1:])(
                model_name=planner_model_name)
        else:
            tools_map[tool_name] = getattr(module, tool_name[0].upper() + tool_name[1:])()
    return tools_map


def load_city(city_set_path: str) -> List[str]:
    city_set = []
    lines = open(city_set_path, 'r').read().strip().split('\n')
    for unit in lines:
        city_set.append(unit)
    return city_set


class FeedbackAgent:
    def __init__(self,
                 args,
                 mode: str = 'zero_shot',
                 tools: List[str] = None,
                 max_steps: int = 30,
                 max_retries: int = 3,
                 illegal_early_stop_patience: int = 3,
                 react_llm_name='gpt-3.5-turbo-1106',
                 planner_llm_name='gpt-3.5-turbo-1106',
                 #  logs_path = '../logs/',
                 city_file_path='D:/semester 7/FYP/agents/planner_agent/database/background/citySet.txt'
                 ) -> None:

        self.finished = None
        self.scratchpad = None
        self.query = None
        self.answer = ''
        self.max_steps = max_steps
        self.mode = mode

        self.react_name = react_llm_name
        self.planner_name = planner_llm_name

        if self.mode == 'zero_shot':
            # TODO: make a prompt for the feedback
            self.agent_prompt = zeroshot_feedback_agent_prompt

        self.json_log = []

        self.current_observation = ''
        self.current_data = None

        if 'gpt-3.5' in react_llm_name:
            stop_list = ['\n']
            self.max_token_length = 15000
            self.llm = ChatOpenAI(temperature=1,
                                  max_tokens=256,
                                  model_name=react_llm_name,
                                  openai_api_key=OPENAI_API_KEY,
                                  model_kwargs={"stop": stop_list})

        elif 'gpt-4' in react_llm_name:
            stop_list = ['\n']
            self.max_token_length = 30000
            self.llm = ChatOpenAI(temperature=0,
                                  max_tokens=256,
                                  model_name=react_llm_name,
                                  openai_api_key=OPENAI_API_KEY,
                                  model_kwargs={"stop": stop_list})

        elif react_llm_name in ['mistral-7B-32K']:
            stop_list = ['\n']
            self.max_token_length = 30000
            self.llm = ChatOpenAI(temperature=0,
                                  max_tokens=256,
                                  openai_api_key="EMPTY",
                                  openai_api_base="http://localhost:8301/v1",
                                  model_name="gpt-3.5-turbo",
                                  model_kwargs={"stop": stop_list})

        elif react_llm_name in ['mixtral']:
            stop_list = ['\n']
            self.max_token_length = 30000
            self.llm = ChatOpenAI(temperature=0,
                                  max_tokens=256,
                                  openai_api_key="EMPTY",
                                  openai_api_base="http://localhost:8501/v1",
                                  model_name="gpt-3.5-turbo",
                                  model_kwargs={"stop": stop_list})

        elif react_llm_name in ['ChatGLM3-6B-32K']:
            stop_list = ['\n']
            self.max_token_length = 30000
            self.llm = ChatOpenAI(
                temperature=0,
                max_tokens=256,
                openai_api_key="EMPTY",
                openai_api_base="http://localhost:8501/v1",
                model_name="gpt-3.5-turbo",
                model_kwargs={"stop": stop_list})

        elif react_llm_name in ['gemini']:
            self.llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-pro", google_api_key=GOOGLE_API_KEY)
            self.max_token_length = 30000

        self.illegal_early_stop_patience = illegal_early_stop_patience

        self.tools = load_tools(tools, planner_model_name=planner_llm_name)
        self.max_retries = max_retries
        self.retry_record = {key: 0 for key in self.tools}
        self.retry_record['invalidAction'] = 0

        # print(self.retry_record)

        self.last_actions = []

        # self.log_path = logs_path + datetime.now().strftime('%Y%m%d%H%M%S') + '.out'
        # self.log_file = open(self.log_path, 'a+')

        # print("logs will be stored in " + self.log_path)

        self.city_set = load_city(city_set_path=city_file_path)

        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

        self.__reset_agent()

    def run(self, user_feedback, original_plan, reset=True) -> tuple[str, Any, list[Any]]:
        """
        Runs the FeedbackAgent to refine the original travel plan based on user feedback.
        """
        self.query = user_feedback
        self.scratchpad = original_plan

        if reset:
            self.__reset_agent()

        while not self.is_halted() and not self.is_finished():
            self.step()

        return self.answer, self.scratchpad, self.json_log

    def step(self) -> None:

        self.json_log.append({"step": self.step_n, "thought": "",
                              "action": "", "observation": "", "state": ""})

        self.scratchpad += f'\nThought {self.step_n}:'
        self.scratchpad += ' ' + self.prompt_agent()

        print(self.scratchpad.split('\n')[-1])
        self.json_log[-1]['thought'] = self.scratchpad.split('\n')[-1].replace(f'\nThought {self.step_n}:', "")
        # self.log_file.write(self.scratchpad.split('\n')[-1] + '\n')

        # Act
        self.scratchpad += f'\nAction {self.step_n}:'
        action = self.prompt_agent()

        if action is None or action == '' or action == '\n':
            self.scratchpad += " Your action is filtered due to content. Please make sure your action does not start with ['\\n', 'Thought', 'Action', 'Observation'] and assume all the actions are permitted in this environment and take the action again."
        else:
            self.scratchpad += ' ' + action

        if len(self.last_actions) > 0 and self.last_actions[-1] != action:
            self.last_actions.clear()

        # refresh last_action list
        self.last_actions.append(action)

        self.json_log[-1]['action'] = self.scratchpad.split('\n')[-1].replace(f'\nAction {self.step_n}:', "")

        # examine if the same action has been repeated 3 times consecutively
        if len(self.last_actions) == 3:
            print("The same action has been repeated 3 times consecutively. So we stop here.")
            # self.log_file.write("The same action has been repeated 3 times consecutively. So we stop here.")
            self.json_log[-1]['state'] = 'same action 3 times repeated'
            self.finished = True
            return

        # action_type, action_arg = parse_action(action)
        print(self.scratchpad.split('\n')[-1])
        # self.log_file.write(self.scratchpad.split('\n')[-1]+'\n')

        # Observe
        self.scratchpad += f'\nObservation {self.step_n}: '

        if action is None or action == '' or action == '\n':
            action_type = None
            action_arg = None
            self.scratchpad += "No feedback from the environment due to the null action. Please make sure your action does not start with [Thought, Action, Observation]."

        else:
            action_type, action_arg = parse_action(action)

            if action_type != "Planner":
                if action_type in actionMapping:
                    pending_action = actionMapping[action_type]
                elif action_type not in actionMapping:
                    pending_action = 'invalidAction'

                if pending_action in self.retry_record:
                    if self.retry_record[pending_action] + 1 > self.max_retries:
                        action_type = 'Planner'
                        print(f"{pending_action} early stop due to {self.max_retries} max retries.")
                        # self.log_file.write(f"{pending_action} early stop due to {self.max_retries} max retries.")
                        self.json_log[-1][
                            'state'] = f"{pending_action} early stop due to {self.max_retries} max retries."
                        self.finished = True
                        return

                elif pending_action not in self.retry_record:
                    if self.retry_record['invalidAction'] + 1 > self.max_retries:
                        action_type = 'Planner'
                        print(f"invalidAction Early stop due to {self.max_retries} max retries.")
                        # self.log_file.write(f"invalidAction early stop due to {self.max_retries} max retries.")
                        self.json_log[-1]['state'] = f"invalidAction early stop due to {self.max_retries} max retries."
                        self.finished = True
                        return

            if action_type == 'FlightSearch':
                try:
                    if validate_date_format(action_arg.split(', ')[2]) and validate_city_format(
                            action_arg.split(', ')[0], self.city_set) and validate_city_format(
                            action_arg.split(', ')[1], self.city_set):
                        self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip(),
                                                                  'Masked due to limited length. Make sure the data has been written in Notebook.')
                        self.current_data = self.tools['flights'].run(action_arg.split(', ')[0],
                                                                      action_arg.split(', ')[1],
                                                                      action_arg.split(', ')[2])
                        self.current_observation = str(to_string(self.current_data))
                        self.scratchpad += self.current_observation
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'

                except DateError:
                    self.retry_record['flights'] += 1
                    self.current_observation = f"'{action_arg.split(', ')[2]}' is not in the format YYYY-MM-DD"
                    self.scratchpad += f"'{action_arg.split(', ')[2]}' is not in the format YYYY-MM-DD"
                    self.json_log[-1]['state'] = f'Illegal args. DateError'

                except ValueError as e:
                    self.retry_record['flights'] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]['state'] = f'Illegal args. City Error'

                except Exception as e:
                    print(e)
                    self.retry_record['flights'] += 1
                    self.current_observation = f'Illegal Flight Search. Please try again.'
                    self.scratchpad += f'Illegal Flight Search. Please try again.'
                    self.json_log[-1]['state'] = f'Illegal args. Other Error'

            elif action_type == 'AttractionSearch':

                try:
                    if validate_city_format(action_arg, self.city_set):
                        self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip().strip(),
                                                                  'Masked due to limited length. Make sure the data has been written in Notebook.')
                        self.current_data = self.tools['attractions'].run(action_arg)
                        self.current_observation = to_string(self.current_data).strip('\n').strip()
                        self.scratchpad += self.current_observation
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'
                except ValueError as e:
                    self.retry_record['attractions'] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]['state'] = f'Illegal args. City Error'
                except Exception as e:
                    print(e)
                    self.retry_record['attractions'] += 1
                    self.current_observation = f'Illegal Attraction Search. Please try again.'
                    self.scratchpad += f'Illegal Attraction Search. Please try again.'
                    self.json_log[-1]['state'] = f'Illegal args. Other Error'

            elif action_type == 'AccommodationSearch':

                try:
                    if validate_city_format(action_arg, self.city_set):
                        self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip().strip(),
                                                                  'Masked due to limited length. Make sure the data has been written in Notebook.')
                        self.current_data = self.tools['accommodations'].run(action_arg)
                        self.current_observation = to_string(self.current_data).strip('\n').strip()
                        self.scratchpad += self.current_observation
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'
                except ValueError as e:
                    self.retry_record['accommodations'] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]['state'] = f'Illegal args. City Error'
                except Exception as e:
                    print(e)
                    self.retry_record['accommodations'] += 1
                    self.current_observation = f'Illegal Accommodation Search. Please try again.'
                    self.scratchpad += f'Illegal Accommodation Search. Please try again.'
                    self.json_log[-1]['state'] = f'Illegal args. Other Error'

            elif action_type == 'RestaurantSearch':

                try:
                    if validate_city_format(action_arg, self.city_set):
                        self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip().strip(),
                                                                  'Masked due to limited length. Make sure the data has been written in Notebook.')
                        self.current_data = self.tools['restaurants'].run(action_arg)
                        self.current_observation = to_string(self.current_data).strip()
                        self.scratchpad += self.current_observation
                        self.__reset_record()
                        self.json_log[-1]['state'] = f'Successful'

                except ValueError as e:
                    self.retry_record['restaurants'] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]['state'] = f'Illegal args. City Error'

                except Exception as e:
                    print(e)
                    self.retry_record['restaurants'] += 1
                    self.current_observation = f'Illegal Restaurant Search. Please try again.'
                    self.scratchpad += f'Illegal Restaurant Search. Please try again.'
                    self.json_log = f'Illegal args. Other Error'

            elif action_type == "CitySearch":
                try:
                    self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip(),
                                                              'Masked due to limited length. Make sure the data has been written in Notebook.')
                    # self.current_data = self.tools['cities'].run(action_arg)
                    self.current_observation = to_string(self.tools['cities'].run(action_arg)).strip()
                    self.scratchpad += self.current_observation
                    self.__reset_record()
                    self.json_log[-1]['state'] = f'Successful'

                except ValueError as e:
                    self.retry_record['cities'] += 1
                    self.current_observation = str(e)
                    self.scratchpad += str(e)
                    self.json_log[-1]['state'] = f'Illegal args. State Error'

                except Exception as e:
                    print(e)
                    self.retry_record['cities'] += 1
                    self.current_observation = f'Illegal City Search. Please try again.'
                    self.scratchpad += f'Illegal City Search. Please try again.'
                    self.json_log = f'Illegal args. Other Error'


            elif action_type == 'GoogleDistanceMatrix':

                try:
                    self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip(),
                                                              'Masked due to limited length. Make sure the data has been written in Notebook.')
                    self.current_data = self.tools['googleDistanceMatrix'].run(action_arg.split(', ')[0],
                                                                               action_arg.split(', ')[1],
                                                                               action_arg.split(', ')[2])
                    self.current_observation = to_string(self.current_data)
                    self.scratchpad += self.current_observation
                    self.__reset_record()
                    self.json_log[-1]['state'] = f'Successful'

                except Exception as e:
                    print(e)
                    self.retry_record['googleDistanceMatrix'] += 1
                    self.current_observation = f'Illegal GoogleDistanceMatrix. Please try again.'
                    self.scratchpad += f'Illegal GoogleDistanceMatrix. Please try again.'
                    self.json_log[-1]['state'] = f'Illegal args. Other Error'


            elif action_type == 'NotebookWrite':
                try:
                    self.scratchpad = self.scratchpad.replace(to_string(self.current_data).strip(),
                                                              'Masked due to limited length. Make sure the data has been written in Notebook.')
                    self.current_observation = str(self.tools['notebook'].write(self.current_data, action_arg))
                    self.scratchpad += self.current_observation
                    self.__reset_record()
                    self.json_log[-1]['state'] = f'Successful'

                except Exception as e:
                    print(e)
                    self.retry_record['notebook'] += 1
                    self.current_observation = f'{e}'
                    self.scratchpad += f'{e}'
                    self.json_log[-1]['state'] = f'Illegal args. Other Error'


            elif action_type == "Planner":
                # try:

                self.current_observation = str(
                    self.tools['planner'].run(str(self.tools['notebook'].list_all()), action_arg))
                self.scratchpad += self.current_observation
                self.answer = self.current_observation
                self.__reset_record()
                self.json_log[-1]['state'] = f'Successful'

            else:
                self.retry_record['invalidAction'] += 1
                self.current_observation = 'Invalid Action. Valid Actions are  FlightSearch[Departure City, Destination City, Date] / ' \
                                           'AccommodationSearch[City] /  RestaurantSearch[City] / NotebookWrite[Short Description] / AttractionSearch[City] / CitySearch[State] / GoogleDistanceMatrix[Origin, Destination, Mode] and Planner[Query].'
                self.scratchpad += self.current_observation
                self.json_log[-1]['state'] = f'invalidAction'

        if action is None or action == '' or action == '\n':
            print(f'Observation {self.step_n}: ' + "No feedback from the environment due to the null action.")
            # write(f'Observation {self.step_n}: ' + "Your action is filtered due to content. Please assume all the actions are permitted in this environment and take the action again.")
            self.json_log[-1]['observation'] = "No feedback from the environment due to the null action."
        else:
            print(f'Observation {self.step_n}: ' + self.current_observation + '\n')
            # rite(f'Observation {self.step_n}: ' + self.current_observation+'\n')
            self.json_log[-1]['observation'] = self.current_observation

        self.step_n += 1

        if action_type and action_type == 'Planner' and self.retry_record['planner'] == 0:
            self.finished = True
            self.answer = self.current_observation
            self.step_n += 1
            return

    def prompt_agent(self) -> str:
        while True:
            try:
                # print(self._build_agent_prompt())
                if self.react_name == 'gemini':
                    request = format_step(self.llm.invoke(self._build_agent_prompt(), stop=['\n']).content)
                else:
                    request = format_step(self.llm([HumanMessage(content=self._build_agent_prompt())]).content)
                # print(request)
                return request
            except:
                catch_openai_api_error()
                print(self._build_agent_prompt())
                print(len(self.enc.encode(self._build_agent_prompt())))
                time.sleep(5)

    def _build_agent_prompt(self) -> str:
        if self.mode == "zero_shot":
            return self.agent_prompt.format(
                query=self.query,
                scratchpad=self.scratchpad)

    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return ((self.step_n > self.max_steps) or (
                len(self.enc.encode(self._build_agent_prompt())) > self.max_token_length)) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.answer = ''
        self.scratchpad: str = ''
        self.__reset_record()
        self.json_log = []
        self.current_observation = ''
        self.current_data = None
        self.last_actions = []

        if 'notebook' in self.tools:
            self.tools['notebook'].reset()

    def __reset_record(self) -> None:
        self.retry_record = {key: 0 for key in self.retry_record}
        self.retry_record['invalidAction'] = 0


### String Stuff ###
gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")


def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)

    try:
        if match:
            action_type = match.group(1)
            action_arg = match.group(2)
            return action_type, action_arg
        else:
            return None, None

    except:
        return None, None


def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')


def truncate_scratchpad(scratch_pad: str, n_tokens: int = 1600, tokenizer=gpt2_enc) -> str:
    lines = scratch_pad.split('\n')
    observations = filter(lambda x: x.startswith('Observation'), lines)
    observations_by_tokens = sorted(observations, key=lambda x: len(tokenizer.encode(x)))
    while len(gpt2_enc.encode('\n'.join(lines))) > n_tokens:
        largest_observation = observations_by_tokens.pop(-1)
        ind = lines.index(largest_observation)
        lines[ind] = largest_observation.split(':')[0] + ': [truncated wikipedia excerpt]'
    return '\n'.join(lines)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the|usd)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def EM(answer, key) -> bool:
    return normalize_answer(str(answer)) == normalize_answer(str(key))

def remove_observation_lines(text, step_n):
    pattern = re.compile(rf'^Observation {step_n}.*', re.MULTILINE)
    return pattern.sub('', text)


def validate_date_format(date_str: str) -> bool:
    pattern = r'^\d{4}-\d{2}-\d{2}$'

    if not re.match(pattern, date_str):
        raise DateError
    return True


def validate_city_format(city_str: str, city_set: list) -> bool:
    if city_str not in city_set:
        raise ValueError(f"{city_str} is not valid city in {str(city_set)}.")
    return True


def parse_args_string(s: str) -> dict:
    # Split the string by commas
    segments = s.split(",")

    # Initialize an empty dictionary to store the results
    result = {}

    for segment in segments:
        # Check for various operators
        if "contains" in segment:
            if "~contains" in segment:
                key, value = segment.split("~contains")
                operator = "~contains"
            else:
                key, value = segment.split("contains")
                operator = "contains"
        elif "<=" in segment:
            key, value = segment.split("<=")
            operator = "<="
        elif ">=" in segment:
            key, value = segment.split(">=")
            operator = ">="
        elif "=" in segment:
            key, value = segment.split("=")
            operator = "="
        else:
            continue  # If no recognized operator is found, skip to the next segment

        # Strip spaces and single quotes
        key = key.strip()
        value = value.strip().strip("'")

        # Store the result with the operator included
        result[key] = (operator, value)

    return result


def to_string(data) -> str:
    if data is not None:
        if type(data) == DataFrame:
            return data.to_string(index=False)
        else:
            return str(data)
    else:
        return str(None)

def merge_updated_plan(original_plan, updated_details):
    """
    Merges the updated details into the original plan while keeping unchanged sections intact.
    """
    # Split original plan into days
    original_lines = original_plan.strip().split("\n")
    updated_lines = updated_details.strip().split("\n")

    merged_plan = []
    updated_map = {}  # Store updates in a dictionary by day

    # Extract updated details as key-value pairs
    current_day = None
    for line in updated_lines:
        line = line.strip()
        if line.lower().startswith("day "):  # Identify day headers
            current_day = line
            updated_map[current_day] = []
        elif current_day:
            updated_map[current_day].append(line)

    # Merge updates while keeping old details where no update exists
    current_day = None
    for line in original_lines:
        stripped_line = line.strip()

        # If we encounter a new day header, check if updates exist
        if stripped_line.lower().startswith("day "):
            current_day = stripped_line
            merged_plan.append(line)  # Keep day header as is
            if current_day in updated_map:
                merged_plan.extend(updated_map[current_day])  # Add updated details
                del updated_map[current_day]  # Remove it from map to avoid duplication
        else:
            if not current_day or current_day not in updated_map:  # Keep unchanged parts
                merged_plan.append(line)

    # If there are any remaining updated sections (newly added days), append them
    for remaining_day, details in updated_map.items():
        merged_plan.append(remaining_day)
        merged_plan.extend(details)

    return "\n".join(merged_plan)

def extract_trip_details(original_plan):
    """
    Parses the provided trip plan text and extracts details like:
    - Origin
    - Destination
    - Transportation
    - Accommodation

    Returns a dictionary with extracted details.
    """
    trip_details = {
        "origin": None,
        "destination": None,
        "transportation": None,
        "accommodation": None
    }

    try:
        # Convert plan to lowercase for case-insensitive matching
        plan_text = original_plan.lower()

        # Extract transportation details
        transport_match = re.search(r"transportation:\s*([^\n]+)", plan_text, re.IGNORECASE)
        if transport_match:
            trip_details["transportation"] = transport_match.group(1).strip()

        # Extract destination (assumes "Current City" is the destination)
        destination_match = re.search(r"current city:\s*([^\n]+)", plan_text, re.IGNORECASE)
        if destination_match:
            trip_details["destination"] = destination_match.group(1).strip()

        # Extract accommodation details
        accommodation_match = re.search(r"accommodation:\s*([^\n]+)", plan_text, re.IGNORECASE)
        if accommodation_match:
            trip_details["accommodation"] = accommodation_match.group(1).strip()

        # If origin is not explicitly mentioned, infer it from the first day's city
        first_day_city_match = re.search(r"day 1:\s*current city:\s*([^\n]+)", plan_text, re.IGNORECASE)
        if first_day_city_match:
            trip_details["origin"] = first_day_city_match.group(1).strip()

    except Exception as e:
        print(f"⚠️ Error extracting trip details: {e}")

    return trip_details

def generate_updated_plan(user_query_: str, original_plan: str):
    tools_list = ["notebook", "flights", "attractions", "accommodations", "restaurants", "googleDistanceMatrix",
                  "planner", "cities"]
    model_name = 'gpt-4-1106-preview'
    agent = FeedbackAgent(None, tools=tools_list, max_steps=30, react_llm_name=model_name,
                          planner_llm_name=model_name)

    # Step 1: Retrieve past trip details if available
    past_trip = extract_trip_details(plan)

    # Step 2: Ask user what they want to modify
    print("\n🔍 What do you want to update?")
    print("1. Transportation")
    print("2. Accommodation")
    print("3. Attractions")
    print("4. Restaurants")
    print("5. All of the above")
    choice = input("Enter the number of your choice: ").strip()

    update_queries = []

    # Step 3: Handle transportation updates
    if choice in ["1", "5"]:
        origin = input(f"Enter departure city (Leave blank to use '{past_trip['origin']}'): ").strip() or past_trip["origin"]
        destination = input(f"Enter destination city (Leave blank to use '{past_trip['destination']}'): ").strip() or past_trip["destination"]
        mode = input("Enter mode of transportation (self-driving/taxi/flight): ").strip()

        if mode.lower() == "flight":
            date = input("Enter travel date (YYYY-MM-DD): ").strip()
            update_queries.append(f"FlightSearch[{origin}, {destination}, {date}]")
        else:
            update_queries.append(f"GoogleDistanceMatrix[{origin}, {destination}, {mode}]")

    # Step 4: Handle accommodation updates
    if choice in ["2", "5"]:
        new_accommodation = input("Enter preferred accommodation city (or leave blank to skip): ").strip()
        if new_accommodation:
            update_queries.append(f"AccommodationSearch[{new_accommodation}]")

    # Step 5: Handle attraction updates
    if choice in ["3", "5"]:
        attraction_city = input("Enter the city for attractions (or leave blank to skip): ").strip()
        if attraction_city:
            update_queries.append(f"AttractionSearch[{attraction_city}]")

    # Step 6: Handle restaurant updates
    if choice in ["4", "5"]:
        restaurant_city = input("Enter the city for restaurants (or leave blank to skip): ").strip()
        if restaurant_city:
            update_queries.append(f"RestaurantSearch[{restaurant_city}]")

    # Step 7: Run the agent with the updates
    if not update_queries:
        print("\n✅ No updates made. Keeping the original plan.")
        return original_plan, None, None

    user_query_ += " " + " ".join(update_queries)
    planner_results, scratch_pad, actions_log = agent.run(user_query_, original_plan)

    # Step 8: Merge the updated details with the original plan
    final_plan = merge_updated_plan(original_plan, planner_results)

    return final_plan, scratch_pad, actions_log


# def generate_updated_plan(user_query_: str,original_plan:str):
#     tools_list = ["notebook", "flights", "attractions", "accommodations", "restaurants", "googleDistanceMatrix",
#                   "planner", "cities"]
#     model_name = 'gpt-4-1106-preview'
#     agent = FeedbackAgent(None, tools=tools_list, max_steps=30, react_llm_name=model_name,
#                        planner_llm_name=model_name)
#
#     # Step1: check for missing trip details
#     print("\n🔍 Checking for previous trip details in Notebook...")
#
#     # Step 2: Retrieve trip info (If missing, ask user)
#     origin = input("Enter departure city (Leave blank if using previous city): ").strip() or "New York"
#     destination = input("Enter destination city (Leave blank if using previous city): ").strip() or "hotel"
#     mode = input("Enter mode of transportation (self-driving/taxi/flight): ").strip()
#
#     # Step 3: Validate inputs before calling APIs
#     if not origin or not destination or not mode:
#         print("⚠️ Missing details! Cannot proceed with transportation update.")
#         return original_plan, None, None  # Keep the old plan if details are missing
#
#     # Step 4: Call the correct API based on mode
#     if mode.lower() == "flight":
#         date = input("Enter travel date (YYYY-MM-DD): ").strip()
#         user_query_ += f" FlightSearch[{origin}, {destination}, {date}]"
#     else:
#         user_query_ += f" GoogleDistanceMatrix[{origin}, {destination}, {mode}]"
#     planner_results, scratch_pad, actions_log = agent.run(user_query_,original_plan)
#
#     return planner_results, scratch_pad, actions_log

if __name__ == '__main__':

    user_query = "I want to plan a weekend trip to NewYork for 2 days"

    plan, scratchpad, action_log = generate_updated_plan(user_query)
    print(plan)