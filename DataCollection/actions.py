from dataclasses import dataclass

@dataclass
class Action:
    action_value: int
    text: str
    audio: str
    image: str

actions = {
    "reset": Action(
        action_value=0,
        text="",
        audio=None,
        image=None
    ),
    "left_elbow_flex": Action(
        action_value=1,
        text="Please flex your left elbow so your arm raises to shoulder level",
        audio="path/to/audio",
        image="path/to/image"
        ),
    "left_elbow_relax": Action(
        action_value=2,
        text="Please relax your left elbow back to original state",
        audio="path/to/audio",
        image="path/to/image"
        ),
    "right_elbow_flex": Action(
        action_value=3,
        text="Please flex your right elbow so your arm raises to shoulder level",
        audio="path/to/audio",
        image="path/to/image"
        ),
    "right_elbow_relax": Action(
        action_value=4,
        text="Please relax your right elbow back to original state",
        audio="path/to/audio",
        image="path/to/image"
        ),
    "end_collection": Action(
        action_value=5,
        text="Data collection ended",
        audio=None,
        image=None
    )
}

procedures = [
    (5, "left_elbow_flex"),
    (10, "left_elbow_relax"),
    (15, "right_elbow_flex"),
    (20, "right_elbow_relax"),
    (25, "end_collection")
]