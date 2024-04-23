class Subject:
    _id = ''
    _visit = 0
    _age = 0

    # Initialize the DataWriter class, with an optional id parameter and an optional visit parameter
    # The id is used to identify the subject whose data is being collected
    # The default value is 0000, the first digit of the id stands for the subject's sex
    # The second digit stands for the subject's group
    # The last two digits are the subject's number in the group
    # The visit parameter is used to identify the visit number of the subject
    def __init__(self, id: str = '0000', visit : int = 0, age: int = 0):
        self._id = id
        self._visit = visit
        self._age = age
        pass

    # Set the _id of the subject
    def set_subject_id(self, id: str):
        self._id = id

    # Set the visit of the subject
    def set_visit(self, visit: int):
        self._visit = visit

    # Get the _id of the subject
    def get_subject_id(self):
        return self._id
    
    # Get the visit of the subject
    def get_visit(self):
        return self._visit

    # Set the age of the subject
    def set_age(self, age: int):
        self._age = age

    # Get the age of the subject
    def get_age(self):
        return self._age

