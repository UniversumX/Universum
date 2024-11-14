from DataCollection.data_collection import experiment_setup

def test_experiment_setup():
    # Call the function with test parameters
    subject_id = "test123"
    visit = 1
    trial = 1
    
    # Run the experiment setup and check for expected properties
    experiment_setup(subject_id, visit, trial)
    assert subject_id == "test123", "Subject ID should match the provided input"
