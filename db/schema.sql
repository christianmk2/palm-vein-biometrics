-- Schema reconstructed from the SQL queries used in palm_vein_gui.py.
-- The original CREATE TABLE statements weren't part of the source code,
-- so this is a best-effort reconstruction based on the columns referenced -
-- adjust types/constraints as needed for your actual setup.

CREATE DATABASE IF NOT EXISTS tracking_db;
USE tracking_db;

CREATE TABLE IF NOT EXISTS persons (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    lastname        VARCHAR(100) NOT NULL,
    firstname       VARCHAR(100) NOT NULL,
    middlename      VARCHAR(100),
    gender          VARCHAR(10),
    contact_number  VARCHAR(20),
    email           VARCHAR(150),
    address         VARCHAR(255),
    province        VARCHAR(100),
    age             VARCHAR(10),
    tracking_id     VARCHAR(20) NOT NULL UNIQUE
);

-- Holding table for an in-progress temperature reading, joined into
-- person_tracks and then deleted (see checkk() / record_temperature_reading()).
CREATE TABLE IF NOT EXISTS temporary (
    id                  INT AUTO_INCREMENT PRIMARY KEY,
    establishment_id    INT NOT NULL,
    temperature         DECIMAL(4,1) NOT NULL,
    tracking_id         VARCHAR(20) NOT NULL
);

CREATE TABLE IF NOT EXISTS person_tracks (
    id                  INT AUTO_INCREMENT PRIMARY KEY,
    person_id           INT NOT NULL,
    establishment_id    INT NOT NULL,
    temperature         DECIMAL(4,1) NOT NULL,
    tracking_id         VARCHAR(20) NOT NULL,
    date_created        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES persons(id)
);
