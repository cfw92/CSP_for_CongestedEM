# !pip install mysql-connector-python

import cochannel_uav_sim, numpy as np, pandas as pd, sqlalchemy as sa
import mysql.connector
from sqlalchemy.engine import URL  

def write_capture_to_mysql(
    x, signals, fs, fc, T,
    mysql_cfg,
    store_components,
    batch_size
):
    """
    Write composite and individual signals to MySQL as timeseries.
    - x: complex composite array (length N)
    - signals: {"ofdm", "gfsk", "uav", "noise"}
    - meta: dict from your build_scene() (must include fs_sps)
    - mysql_cfg: {host, user, password, database}
    """
    N = int(len(x))

    #Establish new connection to the MySQL install
    mydb = mysql.connector.connect(**mysql_cfg)
    cur = mydb.cursor()

    #create a new database
    cur.execute('DROP DATABASE IF EXISTS rf_db')
    cur.execute("CREATE DATABASE rf_db")    

    #connect to the new database
    rf_db = mysql.connector.connect(
    **mysql_cfg,
    database='rf_db'
    )

    rf_db.autocommit = False
    cur = rf_db.cursor()
    
    # Ensure tables exist & establish schema
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rf_capture (
          capture_id       BIGINT AUTO_INCREMENT PRIMARY KEY,
          created_at       TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
          fs_sps           INT NOT NULL,
          duration_s       DOUBLE NOT NULL,
          samples          INT NOT NULL,
          center_frequency DOUBLE NOT NULL
        ) ENGINE=InnoDB;
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rf_sample (
          capture_id  BIGINT NOT NULL,
          signal_name VARCHAR(16) NOT NULL,
          sample_idx  INT NOT NULL,
          i_val       FLOAT NOT NULL,
          q_val       FLOAT NOT NULL,
          PRIMARY KEY (capture_id, signal_name, sample_idx),
          INDEX idx_cap_samp (capture_id, sample_idx),
          CONSTRAINT fk_cap FOREIGN KEY (capture_id)
          REFERENCES rf_capture(capture_id) ON DELETE CASCADE
        ) ENGINE=InnoDB;
    """)
    rf_db.commit()

    # Insert capture header and get capture_id
    cur.execute(
        "INSERT INTO rf_capture (fs_sps, duration_s, samples, center_frequency) VALUES (%s,%s,%s,%s)",
        (int(fs), float(T), int(N), float(fc))
    )
    capture_id = cur.lastrowid

    # Prepared insert for samples
    insert_sql = ("INSERT INTO rf_sample "
                  "(capture_id, signal_name, sample_idx, i_val, q_val) "
                  "VALUES (%s, %s, %s, %s, %s)")

    def stream_signal(name: str, arr: np.ndarray):
        idx = np.arange(arr.shape[0], dtype=np.int64)
        i_val = arr.real.astype(np.float32)
        q_val = arr.imag.astype(np.float32)

        for start in range(0, len(arr), batch_size):
            end = min(start + batch_size, len(arr))
            batch = list(zip(
                [capture_id] * (end - start),
                [name] * (end - start),
                idx[start:end].tolist(),
                i_val[start:end].tolist(),
                q_val[start:end].tolist()
            ))
            cur.executemany(insert_sql, batch)
            rf_db.commit()

    # Composite first
    stream_signal("composite", x)

    # Components (optional)
    if store_components:
        for k in ("ofdm", "gfsk", "uav", "noise"):
            if k in signals:
                stream_signal(k, signals[k])

    cur.close()
    rf_db.close()
    return capture_id

def main():
    # Define parameters
    x, signals, fs, fc, T = cochannel_uav_sim.main()
    store_components=True # set False if you only want the composite
    batch_size=int(50e3)

    # Establish DB config parameters
    mysql_cfg = {
    "host": "localhost",
    "user": "root",
    "password": "P@$$w0rd_123"
    #"database": "rf_db",
    }
                
    # Write to TimeSeries Database
    cap_id = write_capture_to_mysql(
        x, signals, fs, fc, T,
        mysql_cfg,
        store_components, 
        batch_size
    )
    print("capture_id:", cap_id)
    
if __name__ == "__main__":
    main()