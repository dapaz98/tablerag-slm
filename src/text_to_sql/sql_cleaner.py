import re


def clean_sql(
    generated_text
):

    sql = generated_text.strip()

    sql = sql.replace(
        "```sql",
        ""
    )

    sql = sql.replace(
        "```",
        ""
    )

    sql = sql.strip()

    match = re.search(
        r"(SELECT.*)",
        sql,
        re.IGNORECASE | re.DOTALL
    )

    if match:

        return (
            match.group(1)
            .strip()
        )

    return sql