const sqlite3 = require('sqlite3').verbose();

// Set up the SQLite database
const db = new sqlite3.Database('eeg_data_fake.db', (err) => {
  if (err) {
    console.error(err.message);
  } else {
    console.log('Connected to the eeg_data_fake database.');
  }
});

db.serialize(() => {
  db.run(
    `CREATE TABLE IF NOT EXISTS eeg_data (
      timestamp INTEGER PRIMARY KEY,
      channel TEXT,
      value REAL)`
  );
});

// Generate fake EEG data
function generateFakeEEGData() {
  const parietalChannels = ['P3', 'Pz', 'P4'];
  const timestamp = Date.now();

  return parietalChannels.map((channel) => {
    const value = (Math.random() * 200) - 100; // Simulate EEG values between -100 and 100
    return { timestamp, channel, value };
  });
}

// Store the fake EEG data in the database
function storeFakeEEGData() {
  const fakeEEGData = generateFakeEEGData();

  fakeEEGData.forEach((d) => {
    const { timestamp, channel, value } = d;

    db.run(
      `INSERT INTO eeg_data (timestamp, channel, value) VALUES (?, ?, ?)`,
      [timestamp, channel, value],
      (err) => {
        if (err) {
          console.error(err.message);
        }
      }
    );
  });
}

// Store fake EEG data every 100ms
setInterval(storeFakeEEGData, 100);

// Close the database connection when the script is terminated
process.on('SIGINT', () => {
  db.close((err) => {
    if (err) {
      console.error(err.message);
    } else {
      console.log('Closed the eeg_data_fake database connection.');
    }
  });
  process.exit();
});
