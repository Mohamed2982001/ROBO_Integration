"""Quick offline tests for profile extraction (no GPU)."""
from brain.profile_extract import extract_profile_facts


def test_english_name_age():
    assert extract_profile_facts("My name is Mahmoud") == {"name": "Mahmoud"}
    assert extract_profile_facts("I am 25 years old") == {"age": 25}
    assert extract_profile_facts("I'm 28 and my name is Omar")["name"] == "Omar"


def test_arabic():
    f = extract_profile_facts("أنا محمود")
    assert f.get("name") == "Mahmoud"
    f2 = extract_profile_facts("عندي 28 سنة")
    assert f2.get("age") == 28


if __name__ == "__main__":
    test_english_name_age()
    test_arabic()
    print("profile_extract OK")
