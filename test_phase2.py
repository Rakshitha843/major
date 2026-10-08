"""
Phase 2 tests: passwords, database, registration/login logic, and the web API.

Run from the project root:
    python -m unittest discover -s tests -t . -v

Every test uses a temporary database file, so your real data is never touched.
"""

import importlib.util
import sqlite3
import unittest

from backend import auth_service, security
from database import calibration, users
from database import connection as db
from tests.helpers import TempDatabaseTestCase
from utils import config

FASTAPI_AVAILABLE = (
    importlib.util.find_spec("fastapi") is not None
    and importlib.util.find_spec("httpx") is not None
)


class SecurityTests(unittest.TestCase):
    def test_correct_password_verifies(self):
        stored = security.hash_password("correct horse battery")
        self.assertTrue(security.verify_password("correct horse battery", stored))

    def test_wrong_password_fails(self):
        stored = security.hash_password("correct horse battery")
        self.assertFalse(security.verify_password("wrong password!!", stored))

    def test_same_password_gives_different_hashes(self):
        # Because each hash has its own random salt
        self.assertNotEqual(security.hash_password("samepassword1"), security.hash_password("samepassword1"))

    def test_password_is_not_visible_in_hash(self):
        self.assertNotIn("samepassword1", security.hash_password("samepassword1"))

    def test_damaged_hash_returns_false_instead_of_crashing(self):
        for bad in ("", "garbage", "scrypt$x$y", "md5$1$2$3$aa$bb", None):
            self.assertFalse(security.verify_password("anything", bad))

    def test_token_hash_is_stable_and_different_from_token(self):
        token = security.generate_token()
        self.assertEqual(security.hash_token(token), security.hash_token(token))
        self.assertNotEqual(security.hash_token(token), token)
        self.assertNotEqual(security.generate_token(), security.generate_token())

    def test_username_rules(self):
        self.assertEqual(security.validate_username("  ravi_k  "), "ravi_k")
        for bad in ("ab", "x" * 31, "has space", "bad!char", ""):
            with self.assertRaises(security.ValidationError):
                security.validate_username(bad)

    def test_password_rules(self):
        security.validate_password("12345678")
        with self.assertRaises(security.ValidationError):
            security.validate_password("short")
        with self.assertRaises(security.ValidationError):
            security.validate_password("x" * 129)


class DatabaseTests(TempDatabaseTestCase):
    def test_all_tables_are_created(self):
        rows = self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        names = {row["name"] for row in rows}
        for table in ("users", "login_tokens", "calibrations", "monitoring_sessions", "measurements", "alerts"):
            self.assertIn(table, names)

    def test_init_db_twice_is_safe(self):
        db.init_db()
        db.init_db()

    def test_duplicate_username_is_rejected_ignoring_case(self):
        users.create_user(self.connection, "Ravi", "hash")
        with self.assertRaises(users.UsernameTakenError):
            users.create_user(self.connection, "ravi", "hash")

    def test_foreign_keys_are_enforced(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.connection.execute(
                "INSERT INTO monitoring_sessions (user_id, start_time) VALUES (999, 'x')"
            )

    def test_valid_token_finds_user_and_expired_token_does_not(self):
        user_id = users.create_user(self.connection, "ravi", "hash")
        users.create_login_token(self.connection, user_id, "good", hours=1)
        users.create_login_token(self.connection, user_id, "old", hours=-1)
        self.assertEqual(users.get_user_by_token_hash(self.connection, "good")["username"], "ravi")
        self.assertIsNone(users.get_user_by_token_hash(self.connection, "old"))
        self.assertIsNone(users.get_user_by_token_hash(self.connection, "never-existed"))

    def test_deleting_a_user_deletes_their_tokens(self):
        user_id = users.create_user(self.connection, "ravi", "hash")
        users.create_login_token(self.connection, user_id, "t", hours=1)
        self.connection.execute("DELETE FROM users WHERE id = ?", (user_id,))
        self.connection.commit()
        self.assertIsNone(users.get_user_by_token_hash(self.connection, "t"))


class AuthServiceTests(TempDatabaseTestCase):
    def test_register_logs_the_user_in(self):
        user, token = auth_service.register(self.connection, "asha", "password123")
        found = auth_service.get_user_from_token(self.connection, token)
        self.assertEqual(found["username"], "asha")
        self.assertEqual(found["id"], user["id"])

    def test_password_is_not_stored_in_plain_text(self):
        auth_service.register(self.connection, "asha", "password123")
        stored = self.connection.execute("SELECT password_hash FROM users").fetchone()[0]
        self.assertNotIn("password123", stored)

    def test_duplicate_registration_fails(self):
        auth_service.register(self.connection, "asha", "password123")
        with self.assertRaises(users.UsernameTakenError):
            auth_service.register(self.connection, "ASHA", "password456")

    def test_weak_password_creates_no_account(self):
        with self.assertRaises(security.ValidationError):
            auth_service.register(self.connection, "asha", "short")
        count = self.connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        self.assertEqual(count, 0)

    def test_login_success(self):
        auth_service.register(self.connection, "asha", "password123")
        user, token = auth_service.login(self.connection, "Asha", "password123")
        self.assertEqual(user["username"], "asha")
        self.assertIsNotNone(auth_service.get_user_from_token(self.connection, token))

    def test_login_with_wrong_password_or_unknown_user_fails(self):
        auth_service.register(self.connection, "asha", "password123")
        with self.assertRaises(auth_service.InvalidCredentialsError):
            auth_service.login(self.connection, "asha", "wrongpassword")
        with self.assertRaises(auth_service.InvalidCredentialsError):
            auth_service.login(self.connection, "nobody", "password123")
        with self.assertRaises(auth_service.InvalidCredentialsError):
            auth_service.login(self.connection, None, None)

    def test_logout_invalidates_the_token(self):
        _, token = auth_service.register(self.connection, "asha", "password123")
        auth_service.logout(self.connection, token)
        self.assertIsNone(auth_service.get_user_from_token(self.connection, token))
        auth_service.logout(self.connection, token)   # logging out twice must not crash
        auth_service.logout(self.connection, None)

    def test_two_users_stay_separate(self):
        user_a, token_a = auth_service.register(self.connection, "asha", "password123")
        user_b, token_b = auth_service.register(self.connection, "bala", "password456")
        self.assertEqual(auth_service.get_user_from_token(self.connection, token_a)["username"], "asha")
        self.assertEqual(auth_service.get_user_from_token(self.connection, token_b)["username"], "bala")

        # Give only user A a calibration; user B must still show as not calibrated.
        calibration.save_eye_baseline(self.connection, user_a["id"], {"ear_open": 0.30, "ear_closed": 0.10})
        self.assertIsNotNone(calibration.get_calibration(self.connection, user_a["id"]))
        self.assertIsNone(calibration.get_calibration(self.connection, user_b["id"]))

        row_a = auth_service.get_user_from_token(self.connection, token_a)
        row_b = auth_service.get_user_from_token(self.connection, token_b)
        self.assertTrue(auth_service.describe_user(self.connection, row_a)["calibrated"])
        self.assertFalse(auth_service.describe_user(self.connection, row_b)["calibrated"])


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi/httpx not installed")
class AuthApiTests(TempDatabaseTestCase):
    def new_client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        return TestClient(app)   # each client has its own cookie jar = its own browser

    def test_full_login_cycle(self):
        client = self.new_client()
        self.assertEqual(client.get("/api/auth/me").status_code, 401)

        response = client.post("/api/auth/register", json={"username": "asha", "password": "password123"})
        self.assertEqual(response.status_code, 201)
        self.assertIn("httponly", response.headers["set-cookie"].lower())

        me = client.get("/api/auth/me")
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["username"], "asha")
        self.assertFalse(me.json()["calibrated"])

        self.assertEqual(client.post("/api/auth/logout").status_code, 200)
        self.assertEqual(client.get("/api/auth/me").status_code, 401)

        login = client.post("/api/auth/login", json={"username": "asha", "password": "password123"})
        self.assertEqual(login.status_code, 200)
        self.assertEqual(client.get("/api/auth/me").json()["username"], "asha")

    def test_error_responses(self):
        client = self.new_client()
        client.post("/api/auth/register", json={"username": "asha", "password": "password123"})
        other = self.new_client()
        self.assertEqual(other.post("/api/auth/register",
                                    json={"username": "asha", "password": "password999"}).status_code, 409)
        self.assertEqual(other.post("/api/auth/register",
                                    json={"username": "bala", "password": "short"}).status_code, 400)
        self.assertEqual(other.post("/api/auth/login",
                                    json={"username": "asha", "password": "nope-nope"}).status_code, 401)

    def test_two_browsers_see_only_their_own_account(self):
        first, second = self.new_client(), self.new_client()
        first.post("/api/auth/register", json={"username": "asha", "password": "password123"})
        second.post("/api/auth/register", json={"username": "bala", "password": "password456"})
        self.assertEqual(first.get("/api/auth/me").json()["username"], "asha")
        self.assertEqual(second.get("/api/auth/me").json()["username"], "bala")

    def test_health_reports_database_ok(self):
        self.assertEqual(self.new_client().get("/api/health").json()["database"], "ok")


if __name__ == "__main__":
    unittest.main()
