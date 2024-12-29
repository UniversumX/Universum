from dataclasses import dataclass


@dataclass
class Action:
    action_value: int
    text: str
    audio: str
    image: str


actions = {
    "reset": Action(action_value=0, text="", audio=None, image=None),
    "left_elbow_flex": Action(
        action_value=1,
        text="Please flex your left elbow so your arm raises to shoulder level",
        audio="path/to/audio",
        image="path/to/image",
    ),
    "left_elbow_relax": Action(
        action_value=2,
        text="Please relax your left elbow back to original state",
        audio="path/to/audio",
        image="path/to/image",
    ),
    "right_elbow_flex": Action(
        action_value=3,
        text="Please flex your right elbow so your arm raises to shoulder level",
        audio="path/to/audio",
        image="path/to/image",
    ),
    "right_elbow_relax": Action(
        action_value=4,
        text="Please relax your right elbow back to original state",
        audio="path/to/audio",
        image="path/to/image",
    ),
    "end_collection": Action(
        action_value=5, text="Data collection ended", audio=None, image=None
    ),
}


procedures = [
    ## Repetition 1
    (3, "left_elbow_flex"),
    (6, "left_elbow_relax"),
    (9, "right_elbow_flex"),
    (12, "right_elbow_relax"),
    ## Repetition 2
    (15 + 2, "left_elbow_flex"),
    (18 + 2, "left_elbow_relax"),
    (21 + 2, "right_elbow_flex"),
    (24 + 2, "right_elbow_relax"),
    ## Repetition 3
    (27 + 4, "left_elbow_flex"),
    (30 + 4, "left_elbow_relax"),
    (33 + 4, "right_elbow_flex"),
    (36 + 4, "right_elbow_relax"),
    ## Repetition 4
    (39 + 6, "left_elbow_flex"),
    (42 + 6, "left_elbow_relax"),
    (45 + 6, "right_elbow_flex"),
    (48 + 6, "right_elbow_relax"),
    ## End :)
    (51 + 6, "end_collection"),
]
