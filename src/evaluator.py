import re


def extract_number(text):

    m = re.search(r"\d+", text)
    if m:
        return int(m.group())

    return None


def extract_boolean(text):

    text = text.lower()

    if "yes" in text:
        return True

    if "no" in text:
        return False

    return None


def extract_list(text):

    return [x.strip() for x in text.split(",")]


def evaluate_answer(question, answer):

    kind = question["answer_spec"]["kind"]

    if kind == "single_number":

        pred = extract_number(answer)
        truth = question["answer_spec"]["value"]

        return pred == truth


    if kind == "boolean":

        pred = extract_boolean(answer)
        truth = question["answer_spec"]["value"]

        return pred == truth


    if kind == "entity":

        pred = answer.strip()
        truth = question["answer_spec"]["value"]

        return pred.lower() == truth.lower()


    if kind == "entity_list" or kind == "ordered_list":

        pred = extract_list(answer)
        truth = question["answer_spec"]["value"]

        return pred == truth


    if kind == "multi_field":

        pred = extract_list(answer)

        year_truth = question["answer_spec"]["fields"][0]["value"]
        name_truth = question["answer_spec"]["fields"][1]["value"]

        return str(year_truth) in pred[0] and name_truth.lower() in pred[1].lower()

    return False