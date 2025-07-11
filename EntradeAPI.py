import requests, os
from dotenv import load_dotenv
from time import sleep
# Phải có 1 file .env trong cùng thư mục với file python này chứa mục "usernameDNSE" và "password"

load_dotenv()
username = os.getenv("usernameDNSE") # Email hoặc số điện thoại đăng kí tài khoản
password = os.getenv("password") # Mật khẩu đăng nhập tài khoản

REAL_BASE_URL = "https://services.entrade.com.vn/smart-order"
DEMO_BASE_URL = "https://services.entrade.com.vn/papertrade-smart-order"

class EntradeAPI:
    def __init__(self, environment="demo"): # Switch to "real" để dùng tiền thật
        self.base_url = DEMO_BASE_URL
        self.environment = environment
        self.token = None
        self.trading_token = None

    def authenticate(self, username, password):
        # Sử dụng URL đăng nhập chung cho cả hai môi trường
        url = "https://services.entrade.com.vn/entrade-api/v2/auth"
        headers = {
            "accept": "*/*",
            "content-type": "application/json"
        }
        data = {
            "username": username,
            "password": password
        }
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        self.token = response.json().get("token")
        print("Authentication successful!\n")

    def get_otp(self):
        if self.environment == "demo":
            print("Demo environment: Skipping OTP step.")
            return

        url = f"{self.base_url}/otp"
        headers = {
            "accept": "*/*",
            "authorization": f"Bearer {self.token}"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        print(response.json()) # Expected: {"status": "OK"}

    def get_investor_info(self):
        url = f"https://services.entrade.com.vn/{"papertrade-" if self.environment == "demo" else ""}entrade-api/investors/_me"
        headers = {
            "accept": "*/*",
            "authorization": f"Bearer {self.token}"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        investor_info = response.json()
        print("Investor info:", investor_info, "\n")
        return investor_info

    def get_account_balance_info(self, investor_id):
        url = f"https://services.entrade.com.vn/{"papertrade-" if self.environment == "demo" else ""}entrade-api/account_balances/{investor_id}"
        headers = {
            "authorization": f"Bearer {self.token}"
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        print("Lấy thông tin tài khoản thành công: ", response.json(), '\n')
        return response.json()

    def get_trading_token(self, otp_code):
        url = f"https://services.entrade.com.vn/entrade-api/otp/trading-token"
        headers = {
            "accept": "*/*",
            "authorization": f"Bearer {self.token}",
            "otp": otp_code
        }
        response = requests.get(url, headers=headers)
        self.trading_token = response.json().get("tradingToken")
        print("Trading token obtained")

    # 4.X SECTIONS
    def get_derivative_info(self):
        url = f"https://services.entrade.com.vn/{"papertrade-" if self.environment == "demo" else ""}entrade-api/derivatives"
        headers = {
            "accept": "*/*",
            "authorization": f"Bearer {self.token}"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        try:
            derivatives = response.json()
            print("Derivatives info:", derivatives,"\n")
            return derivatives
        except requests.exceptions.JSONDecodeError:
            print("Error: Unable to decode JSON response")
            print("Response content:", response.text)
            return None
        

    def place_conditional_order(self, portfolio_id, condition, expired_time, investor_account_id, investor_id, symbol, price, quantity, side="NB"):
        url = f"{self.base_url}/orders"
        headers = {
            "accept": "*/*",
            "content-type": "application/json",
            "authorization": f"Bearer {self.token}",
            "trading-token": self.trading_token
        }

        data = {
            "investorId": investor_id,
            "investorAccountId": investor_account_id, # MUST BE INT
            "bankMarginPortfolioId": portfolio_id,
            "expiredTime": expired_time,
            "symbol": symbol,
            "targetPrice": price,
            "targetSide": side,
            "type": "STOP",
            "targetQuantity": quantity,
            "condition": f"price {condition}"
        }

        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        print("Đặt lệnh điều kiện thành công:", response.json(), '\n')

        return response.json()

    def get_conditional_orders(self, investor_id, amount = 1000):
        url = f"{self.base_url}/orders"
        params = {
            "investorId": investor_id,
            "_order": "DESC", # Or ASC?
            "_sort": "createdDate",
            "_start": 0,
            "_end": amount
        }
        headers = {
            "accept": "*/*",
            "authorization": f"Bearer {self.token}",
        }

        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()

        print("Lấy danh sách lệnh thành công:", response.json())
        return response.json()["data"]

    def cancel_conditional_order(self, order_id):
        url = f"{self.base_url}/orders/{order_id}"
        headers = {
            "accept": "*/*",
            "authorization": f"Bearer {self.token}",
            "trading-token": self.trading_token
        }

        response = requests.delete(url, headers=headers)
        response.raise_for_status()

        print(f"Hủy lệnh {order_id} thành công:", response.json())
        return response.json()

    def cancel_all_conditional_orders(self, investor_account_id, investor_id):
        url = f"{self.base_url}/orders"
        params = {
            "investorAccountId": investor_account_id,
            "investorId": investor_id
        }

        headers = {
            "accept": "*/*",
            "authorization": f"Bearer {self.token}",
            "trading-token": self.trading_token
        }

        response = requests.delete(url, params=params, headers=headers)
        response.raise_for_status()

        print("Hủy tất cả lệnh thành công:", response.json()) # Expected: {"status": "OK"}

    # 5.X SECTIONS
    def secure_deal(self, amount, deal_id, investor_account_id, investor_id, transaction_type = "CASH_DEPOSIT_DEAL_SECURE"):
        url = f"https://services.entrade.com.vn/{"papertrade-" if self.environment == "demo" else ""}entrade-api/transactions"
        headers = {
            "accept": "*/*",
            "authorization": f"Bearer {self.token}"
        }

        data = {
            "investorId": investor_id,
            "investorAccountId": investor_account_id,
            "data": {
                "investorId": investor_id,
                "investorAccountId": investor_account_id, # MUST BE INT
                "amount": amount, # MUST BE INT
                "dealId": deal_id # MUST BE INT
            },
            "transactionType": transaction_type
        }

        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()

        print("Nộp cọc thành công: ", response.json())
        return response.json()

    def boost_purchase_power(self, amount, investor_account_id):
        url = f"https://services.entrade.com.vn/{"papertrade-" if self.environment == "demo" else ""}entrade-api/transactions/advance-vsd-secure"
        headers = {
            "accept": "*/*",
            "authorization": f"Bearer {self.token}"
        }

        data = {
            "amount": amount, # MUST BE INT
            "investorAccountId": investor_account_id # MUST BE INT
        }

        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()

        print("Tăng sức mua thành công: ", response.json())
        return response.json()

    # 6.X SECTIONS
    def get_strategies(self, investor_id, amount = 100):
        url = "https://services.entrade.com.vn/bot-api/strategies"
        headers = {
            "accept": "*/*",
            "authorization": f"Bearer {self.token}"
        }

        params = {
            "_end": amount,
            "_start": 0,
            "investorId": investor_id
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        print("Lấy danh sách chiến lược thành công:", response.json(), '\n')
        return response.json()

    def get_strategies_simplified(self, investor_id): 
        url = f"https://services.entrade.com.vn/bot-api/strategies/group"
        headers = {
            "accept": "*/*",
            "authorization": f"Bearer {self.token}"
        }

        params = {
            "investorId": investor_id
        }

        response = requests.get(url=url, headers=headers, params=params)
        response.raise_for_status()
        print("Lấy danh sách chiến lược thành công:", response.json()["data"], '\n')
        return response.json()["data"]

    def add_fund_for_bot(self, investor_id, investor_account_id, bot_fund_account_id, amount = 5000000):
        url = f"https://services.entrade.com.vn/{"papertrade-" if self.environment == "demo" else ""}entrade-api/transactions"
        headers = {
            "accept": "*/*",
            "authorization": f"Bearer {self.token}"
        }

        data = {
            "investorId": investor_id,
            "investorAccountId": investor_account_id,
            "data": {
                "investorId": investor_id,
                "investorAccountId": investor_account_id,
                "toInvestorAccountId": bot_fund_account_id,
                "amount": amount
            },
            "transactionType": "CASH_SUB_TRANSFER_SENT"
        }

        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        print("Nạp tiền cho bot thành công: ", response.json(), '\n')
        return response.json()

    def withdrawn_fund_from_bot(self, investor_id, investor_account_id, bot_fund_account_id, amount = 5000000):
        url = f"https://services.entrade.com.vn/{"papertrade-" if self.environment == "demo" else ""}entrade-api/transactions"
        headers = {
            "accept": "*/*",
            "authorization": f"Bearer {self.token}"
        }

        data = {
            "investorId": investor_id,
            "investorAccountId": bot_fund_account_id,
            "data": {
                "investorId": investor_id,
                "investorAccountId": bot_fund_account_id,
                "toInvestorAccountId": investor_account_id,
                "amount": amount
            },
            "transactionType": "CASH_SUB_TRANSFER_SENT"
        }
        
        response = requests.post(url=url, json=data, headers=headers)
        response.raise_for_status()
        print("Rút tiền từ bot thành công: ", response.json(), '\n')

    def get_bots(self, investor_id, status="PENDING,FAILED,STOPPED", amount = 100):
        url = f"https://services.entrade.com.vn/bot-api/bots"
        headers = {
            "accept": "*/*",
            "authorization": f"Bearer {self.token}",
        }

        params = {
            "_start": 0,
            "_end": amount,
            "investorId": investor_id,
            "status": status
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        print("Lấy danh sách bot thành công:", response.json())
        return response.json()

    def activate_bot(self, env, strategy_id, strategy_version, loss_limit = 1000000):
        url = f"https://services.entrade.com.vn/{"papertrade-" if self.environment == "demo" else ""}bot-api/bots"
        headers = {
            "accept": "*/*",
            "authorization": f"Bearer {self.token}",
            "trading-token": self.trading_token
        }

        data = {
            "env": env,
            "strategyId": strategy_id,
            "strategyVersion": strategy_version,
            "lossLimit": loss_limit
        }

        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        print("Triển khai bot thành công:", response.json())
        return response.json()

    def activate_and_add_funds_for_bot(self, investor_id, investor_account_id, allocated_amount, env, strategy_id, strategy_version, loss_limit = 1000000):
        botAccount = self.activate_bot(env, strategy_id, strategy_version, loss_limit)
        self.add_fund_for_bot(investor_id, investor_account_id, botAccount["investorAccountId"], allocated_amount)
        return botAccount["id"]

    def stop_bot(self, bot_id):
        url = f"https://services.entrade.com.vn/{"papertrade-" if self.environment == "demo" else ""}bot-api/bots/{bot_id}"
        headers = {
            "accept": "*/*",
            "authorization": f"Bearer {self.token}",
            "trading-token": self.trading_token
        }

        response = requests.delete(url, headers=headers)
        response.raise_for_status()
        print("Dừng bot thành công:", response.json())
        return response.json()

    def get_backtest_list(self, investor_id, amount = 100):
        url = f"https://services.entrade.com.vn/bot-api/backtests"
        headers = {
            "accept": "*/*",
            "authorization": f"Bearer {self.token}"
        }

        param = {
            "_start": 0,
            "_end": amount,
            "investorId": investor_id
        }

        response = requests.get(url=url, headers=headers, params=param)
        response.raise_for_status()
        print("Lấy danh sách backtest thành công:", response.json()['data'], '\n')
        return response.json()['data']

    def delete_backtest(self, backtest_id):
        url = f"https://services.entrade.com.vn/bot-api/backtests/{backtest_id}"
        headers = {
            "accept": "*/*",
            "authorization": f"Bearer {self.token}"
        }

        response = requests.delete(url=url, headers=headers)
        response.raise_for_status()

        print("Xóa backtest thành công:", response.json(), '\n')
        return response.json()

if __name__ == "__main__":
    client = EntradeAPI(environment="demo") # thay đổi thành "real" cho môi trường tiền thật, "demo" cho môi trường tiền ảo

    client.authenticate(username, password)
    otp_code = None
    if client.environment == "real":
        client.get_otp()
        otp_code = input("Enter OTP: ")
        client.get_trading_token(otp_code)

    investor_info = client.get_investor_info()
    investor_id = investor_info["investorId"]
    investor_account_id = client.get_account_balance_info(investor_id)["investorAccountId"]

    # client.get_derivative_info()

    # ĐẶT LỆNH ĐIỀU KIỆN
    # order_id = client.place_conditional_order(32, ">= 1392", "2025-03-27T07:30:00.000Z", investor_account_id, investor_id, "VN30F2504", 1392, 1, "NS")["id"]

    # sleep(10)

    # client.cancel_conditional_order(order_id)
    # client.cancel_all_conditional_orders(investor_account_id, investor_id)

    # client.get_conditional_orders(investor_id)


    # ADD BOT AVATAR

    # backtest_list = client.get_backtest_list(investor_id)
    # backtest = backtest_list[1]
    # strategy_id, strategy_version = backtest["strategyId"], backtest["strategyVersion"]


    # strategies = client.get_strategies(investor_id)["data"]
    # strategy = strategies[1]
    # strategy_id, strategy_version = strategy["id"], strategy["version"]


    # strategies = client.get_strategies_simplified(investor_id)
    # strategy = strategies[4]
    # strategy_id, strategy_version = strategy["id"], strategy["versions"][0]

    # bot_id = client.activate_and_add_funds_for_bot(investor_id, investor_account_id, 17072004, "papertrade", strategy_id, strategy_version)

    # sleep(10)
    # client.stop_bot(bot_id)

    # CHECK, XOÁ BACKTEST
    # backtest_list = client.get_backtest_list(investor_id)
    # backtest = backtest_list[0]

    # client.delete_backtest(backtest["id"])