from enum import Enum

RAW_FILE_DIR = "../data/raw/"
PROCESSED_FILE_DIR = "../data/processed/"

RAW_COMPLAINTS_DATA_FILE_NAME = "complaints.csv"
CLENAED_COMPLAINTS_DATA_FILE_NAME = "complaints_clean.csv"


class Columns(Enum):
    DATE_RECEIVED = "Date received"
    PRODUCT = "Product"
    SUB_PRODUCT = "Sub-product"
    ISSUE = "Issue"
    SUB_ISSUE = "Sub-issue"
    COMPLAINT = "Consumer complaint narrative"
    COMPANY_PUBLIC_RESPONSE = "Company public response"
    COMPANY = "Company"
    STATE = "State"
    ZIP_CODE = "ZIP code"
    TAGS = "Tags"
    CONSUMER_CONSENT = "Consumer consent provided?"
    SUBMITTED_VIA = "Submitted via"
    DATE_SENT_TO_COMPANY = "Date sent to company"
    COMPANY_RESPONSE_TO_CONSUMER = "Company response to consumer"
    TIMELY_RESPONSE = "Timely response?"
    CONSUMER_DISPUTED = "Consumer disputed?"
    COMPLAINT_ID = "Complaint ID"


class Processed_Columns(Enum):
    WORD_COUNT = "Word Count"


date_columns = [Columns.DATE_RECEIVED.value, Columns.DATE_SENT_TO_COMPANY.value]

product_categories = [
    "Credit card",
    "Payday loan, title loan, or personal loan",
    "Checking or Savings account",
    "Money transfers",
]
