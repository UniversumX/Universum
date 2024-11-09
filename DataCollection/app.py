import os
from sqlalchemy import create_engine, text, inspect
from dotenv import load_dotenv


load_dotenv()
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')

print(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

engine = create_engine(
    f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')


def create_tables():
    with engine.connect() as connection:
        with connection.begin():
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS accelerometer_data (
                    timestamp TIMESTAMPTZ NOT NULL,
                    device_id VARCHAR(255) NOT NULL,
                    x FLOAT,
                    y FLOAT,
                    z FLOAT,
                    pitch FLOAT,
                    roll FLOAT,
                    acceleration FLOAT,
                    inclination FLOAT,
                    patient_id VARCHAR(50) NOT NULL,
                    visit_number INT NOT NULL,
                    trial_number INT NOT NULL
                );
            """))
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS action_data (
                    timestamp TIMESTAMPTZ NOT NULL,
                    action_value INT NOT NULL,
                    patient_id VARCHAR(50) NOT NULL,
                    visit_number INT NOT NULL,
                    trial_number INT NOT NULL
                );
            """))
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS eeg_data_raw (
                    timestamp TIMESTAMPTZ NOT NULL,
                    cp3 FLOAT,
                    c3 FLOAT,
                    f5 FLOAT,
                    po3 FLOAT,
                    po4 FLOAT,
                    f6 FLOAT,
                    c4 FLOAT,
                    cp4 FLOAT,
                    patient_id VARCHAR(50) NOT NULL,
                    visit_number INT NOT NULL,
                    trial_number INT NOT NULL
                );
            """))
        print("Tables created (if they didn't already exist).")


def push_accelerometer_data(data):
    try:
        with engine.connect() as connection:
            connection.execute("""
                INSERT INTO accelerometer_data (timestamp, device_id, x, y, z, pitch, roll, acceleration, inclination, patient_id, visit_number)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, data)
            print("Data pushed to accelerometer_data table.")
    except Exception as e:
        print(f"Error while pushing action data: {e}")


def push_action_data(data):
    try:
        with engine.connect() as connection:
            connection.execute("""
                INSERT INTO action_data (timestamp, action_value, patient_id, visit_number)
                VALUES (%s, %s, %s, %s);
            """, data)
            print("Data pushed to action_data table.")
    except Exception as e:
        print(f"Error while pushing action data: {e}")


def push_eeg_data(data):
    try:
        with engine.connect() as connection:
            connection.execute("""
                INSERT INTO eeg_data_raw (timestamp, cp3, c3, f5, po3, po4, f6, c4, cp4, patient_id, visit_number)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """, data)
            print("Data pushed to eeg_data_raw table.")
    except Exception as e:
        print(f"Error while pushing action data: {e}")


if __name__ == "__main__":
    create_tables()
    inspector = inspect(engine)
    print(inspector.get_table_names())
