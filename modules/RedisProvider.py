# from upstash_redis import Redis
import redis
import ssl

class Cache:

    def __init__(self, host, password, port=34609) -> None:
        self.conn = redis.StrictRedis(
            host=host,
            port=port,
            password=password,
            ssl=True,
            ssl_cert_reqs=None,  # Disables certificate verification
            ssl_ca_certs=None,   # Disables loading CA certificates
            ssl_keyfile=None,    # No client SSL key specified
            ssl_certfile=None    # No client SSL certificate specified
        )

    def get(self, key):
        return self.conn.get(key)
    
    def set(self, key, value):
        return self.conn.set(key, value)
    
    def delete(self, key):
        return self.conn.delete(key)
    
    def get_messages(self, user_id):
        user_records = self.conn.keys(f'*{user_id}*')
        user_messages = [
            {'key': key.decode(), 'messages': [message.decode() for message in self.conn.lrange(key, 0, -1)][::-1]} 
            for key in user_records
        ]

        return user_messages