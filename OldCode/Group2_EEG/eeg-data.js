const { Notion } = require('@neurosity/notion');
const sqlite3 = require('sqlite3').verbose();

// Initialize the Neurosity Notion device
const notion = new Notion({
  deviceId: 'your-device-id',
  email: 'your-email',
  password: 'your-password',
});

// Set up the SQLite database
const db = new sqlite3.Database('eeg_data.db', (err) => {
  if (err) {
    console.error(err.message);
  } else {
    console.log('Connected to the eeg_data database.');
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

// Connect to the Notion device
notion
  .login()
  .then(() => notion.selectDevice())
  .then(() => {
    console.log('Connected to the Notion device.');

    // Subscribe to EEG data
    notion.eeg().subscribe((eegData) => {
      // Parietal lobe channels: P3, Pz, P4
      const parietalChannels = ['P3', 'Pz', 'P4'];
      const parietalData = eegData.data.filter((d) => parietalChannels.includes(d.label));

      // Store the parietal lobe EEG data in the database
      parietalData.forEach((d) => {
        const timestamp = eegData.timestamp;
        const channel = d.label;
        const value = d.value;

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
    });
  })
  .catch((error) => {
    console.error(error);
  });

// Close the database connection when the script is terminated
process.on('SIGINT', () => {
  db.close((err) => {
    if (err) {
      console.error(err.message);
    } else {
      console.log('Closed the eeg_data database connection.');
    }
  });
  process.exit();
});
