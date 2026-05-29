import sqlite3
import pandas as pd
from datetime import datetime
from config import DB_PATH


class TitanicDatabase:
    def __init__(self):
        self.conn = None
        self._init_database()

    def _init_database(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS passengers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pclass INTEGER NOT NULL,
                name TEXT NOT NULL,
                sex TEXT NOT NULL,
                age REAL,
                sibsp INTEGER,
                parch INTEGER,
                ticket TEXT,
                fare REAL,
                cabin TEXT,
                embarked TEXT,
                survival_probability REAL NOT NULL,
                predicted_survived INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()

    def add_passenger(self, passenger_data, survival_probability):
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT INTO passengers 
            (pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked, survival_probability, predicted_survived)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            passenger_data.get('Pclass'),
            passenger_data.get('Name'),
            passenger_data.get('Sex'),
            passenger_data.get('Age'),
            passenger_data.get('SibSp'),
            passenger_data.get('Parch'),
            passenger_data.get('Ticket', ''),
            passenger_data.get('Fare'),
            passenger_data.get('Cabin', ''),
            passenger_data.get('Embarked'),
            survival_probability,
            1 if survival_probability >= 0.5 else 0
        ))
        
        self.conn.commit()
        return cursor.lastrowid

    def get_all_passengers(self):
        df = pd.read_sql_query('SELECT * FROM passengers ORDER BY created_at DESC', self.conn)
        return df

    def get_survived_passengers(self):
        df = pd.read_sql_query(
            'SELECT * FROM passengers WHERE predicted_survived = 1 ORDER BY created_at DESC',
            self.conn
        )
        return df

    def get_passenger_by_id(self, passenger_id):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM passengers WHERE id = ?', (passenger_id,))
        return cursor.fetchone()

    def delete_passenger(self, passenger_id):
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM passengers WHERE id = ?', (passenger_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def get_statistics(self):
        cursor = self.conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM passengers')
        total = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM passengers WHERE predicted_survived = 1')
        survived = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(survival_probability) FROM passengers')
        avg_prob = cursor.fetchone()[0]
        
        return {
            'total': total,
            'survived': survived,
            'not_survived': total - survived,
            'avg_survival_probability': avg_prob if avg_prob else 0
        }

    def close(self):
        if self.conn:
            self.conn.close()
