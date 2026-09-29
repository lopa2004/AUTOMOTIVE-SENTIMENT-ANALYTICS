import re


ASPECT_KEYWORDS = {

    "Vehicle": [
        "car", "vehicle", "model", "design", "body", "build"
    ],

    "Engine": [
        "engine", "motor", "power", "horsepower",
        "acceleration", "performance"
    ],

    "Battery": [
        "battery", "range", "charging",
        "charge", "battery life"
    ],

    "Mileage": [
        "mileage", "fuel economy", "consumption",
        "kmpl", "efficiency", "fuel efficient"
    ],

    "Safety": [
        "safety", "airbag", "brake", "braking",
        "abs", "adas", "collision", "protection"
    ],

    "Comfort": [
        "comfort", "comfortable", "seat", "seats",
        "ride", "suspension", "legroom", "cabin"
    ],

    "Service": [
        "service", "maintenance", "dealer",
        "repair", "workshop"
    ],

    "Infotainment": [
        "infotainment", "screen", "display",
        "navigation", "audio", "speaker",
        "bluetooth", "touchscreen"
    ],

    "Price": [
        "price", "cost", "expensive",
        "affordable", "value", "priced"
    ]
}


def extract_aspects(text):

    text = text.lower()

    found_aspects = []

    for aspect, keywords in ASPECT_KEYWORDS.items():

        for keyword in keywords:

            pattern = r"\b" + re.escape(keyword) + r"\b"

            if re.search(pattern, text):

                found_aspects.append(aspect)

                break

    return found_aspects


if __name__ == "__main__":

    test_reviews = [

        "The battery range is excellent.",

        "The engine is powerful but the mileage is poor.",

        "The seats are comfortable and the infotainment system is easy to use.",

        "The car is expensive but the safety features are excellent."

    ]

    for review in test_reviews:

        aspects = extract_aspects(review)

        print("\nReview:", review)
        print("Detected aspects:", aspects)