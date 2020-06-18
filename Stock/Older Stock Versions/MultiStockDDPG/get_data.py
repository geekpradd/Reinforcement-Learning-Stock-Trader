import requests

def download(url, location):
    r = requests.get(url, allow_redirects=True)
    open(location, 'wb').write(r.content)

COMPONENTS = ['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DOW', 'XOM', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'RTX', 'TRV', 'UNH', 'VZ', 'V', 'WMT', 'WBA', 'DIS']

INDEX = "%5EDJI"
url = "https://query1.finance.yahoo.com/v7/finance/download/{0}?period1={1}&period2={2}&interval=1d&events=history"

train_start = "1230768000"
train_end = "1451606400"
test_start = "1451606400"
test_end = "1538265600"

#for component in COMPONENTS:
#    print ("Downloading " + component)
download(url.format(INDEX, test_start, test_end), "index/{0}-test.csv".format(INDEX))
download(url.format(INDEX, train_start, train_end), "index/{0}-train.csv".format(INDEX))




