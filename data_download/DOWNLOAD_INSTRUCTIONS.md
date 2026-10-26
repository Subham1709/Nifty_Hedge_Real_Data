# Getting the NSE Nifty option data

The goal is NSE's daily F&O bhavcopy files from Jan 2020 to Apr 2026, filtered to Nifty options and futures. They have to be downloaded on your side because NSE is blocked from Claude's cloud machine.

## Route 1: Google Colab (try this first)

1. Open `nse_fo_download_colab.ipynb` in Colab (File > Upload notebook).
2. Click Runtime > Run all and allow Google Drive access.
3. Step 3 runs a quick test. If it says `All good`, leave it running for 30 to 60 minutes.
4. If some days fail, run Step 4 again. It only fetches what is missing.
5. Upload the files from `My Drive/nse_fo_nifty/`: `nifty_fo_2020.parquet` to `nifty_fo_2026.parquet` and `download_log.csv`. You can also upload the single zip, `nifty_fo_bhavcopy_2020_2026.zip`.

## Route 2: your laptop (if Colab prints BLOCKED)

NSE sometimes refuses Google's servers but almost never refuses a home connection in India.

1. Install Python 3.10 or newer from python.org if you don't have it.
2. Put `nse_fo_download.py` in a folder, open a terminal in that folder and run:
   ```
   pip install pandas pyarrow requests
   python nse_fo_download.py
   ```
3. It creates a `nse_fo_nifty` folder with the same files as Route 1. If it stops part way, run it again and it resumes.

Claude can also run this on your laptop through the device link, if you approve it.

## Route 3: manual, only for a few missing days

On nseindia.com, go to Market Data > All Reports > Derivatives, pick the date and download "F&O - Bhavcopy". Send the zip files as they are, without unzipping or renaming them. This is one day per click, so use it only to fill gaps the log shows.

## Uploading

Upload the files in this thread. If a file is too large to upload, put the files in a `data/` folder in the new GitHub repo instead, and Claude will pick them up from there.

## What is kept

- Nifty index options with expiry within 200 days. Puts are kept from 65% to 110% of the futures price, and calls from 90% to 115%.
- Nifty futures (all expiries).
- Strikes with zero open interest and zero volume are dropped.
- Columns: date, expiry, strike, option type, open/high/low/close, settlement price, volume, turnover, open interest, and the underlying price (in the newer files).
