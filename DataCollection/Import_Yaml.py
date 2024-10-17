import yaml
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class Action:
    action_value: int
    text: str
    audio: str
    image: str

@dataclass
class Study:
    study_name: str
    study_description: str
    actions: Dict[str, Action]
    procedure: List[Tuple[str, str]]

def load_study_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Transform procedure into a list of tuples
    procedure = [(entry['timestamp'], entry['action']) for entry in config['procedure']]

    # Transform actions into Action instances
    actions = {
        action_name: Action(
            action_value=action_data['action_value'],
            text=action_data['text'],
            audio=action_data['audio'],
            image=action_data['image']
        ) for action_name, action_data in config['actions'].items()
    }

    # Create a Study instance
    study = Study(
        study_name=config['study_name'],
        study_description=config['study_description'],
        actions=actions,
        procedure=procedure
    )
    
    return study



# Example usage
if __name__ == "__main__":
    config_path = 'Study_Config.yml'
    study_config = load_study_config(config_path)
    print(study_config.study_name)
    print(study_config.study_description)
    print(study_config.procedure)
    print("=======================")
    for action, data in study_config.actions.items():
        print(action)
        print(data.action_value)
        print(data.text)
        print(data.audio)
        print(data.text)
        print("-------")
    # print(study_config.actions.items())
