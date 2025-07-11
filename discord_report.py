import aiohttp

class DiscordReporter:
    """
    Sends messages to a Discord channel via webhook.
    """
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.last_report_time = 0

    async def send_message(self, message: str):
        payload = {"content": message}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as resp:
                    if resp.status in (200, 204):
                        print(f"[DISCORD] Sent: {message}")
                    else:
                        txt = await resp.text()
                        print(f"[DISCORD ERROR] {resp.status}: {txt}")
        except Exception as e:
            print(f"[DISCORD ERROR] {e}")
