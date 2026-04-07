"""
Synthetic email dataset for EmailTriageEnv.
Covers a realistic corporate inbox with varied categories, urgency, and reply needs.
"""

EMAILS = [
    {
        "id": "e001",
        "from": "sarah.johnson@client-acme.com",
        "to": "support@company.com",
        "subject": "URGENT: Production server down - losing revenue",
        "body": (
            "Hi team,\n\nOur production environment has been completely down for the past 2 hours. "
            "We are a paying customer on the Enterprise plan. This is causing us significant revenue loss "
            "— approximately $10,000/hour. We need immediate assistance.\n\n"
            "Error: 503 Service Unavailable on all endpoints.\n"
            "Account ID: ENT-4892\n\nPlease escalate immediately.\n\nSarah Johnson\nCTO, ACME Corp"
        ),
        "timestamp": "2024-01-15T09:03:00Z",
        "has_attachment": False,
        "correct_category": "support",
        "correct_priority": "critical",
        "correct_department": "engineering",
        "requires_reply": True,
        "reply_key_points": [
            "acknowledge urgency",
            "apologize for downtime",
            "confirm escalation to engineering",
            "provide ticket number or ETA",
        ],
        "reply_tone": "professional",
        "reply_policy": ["do not promise specific uptime SLA without engineering confirmation"],
    },
    {
        "id": "e002",
        "from": "newsletter@marketingplatform.io",
        "to": "support@company.com",
        "subject": "Your weekly growth digest — 5 tactics to 10x conversions",
        "body": (
            "Hi there! This week's digest covers:\n"
            "• 5 proven tactics to increase your conversion rate\n"
            "• Case study: How Company X grew 300% using email automation\n"
            "• Free webinar Thursday 2pm EST\n\n"
            "Click here to unsubscribe."
        ),
        "timestamp": "2024-01-15T08:45:00Z",
        "has_attachment": False,
        "correct_category": "spam",
        "correct_priority": "low",
        "correct_department": None,
        "requires_reply": False,
        "reply_key_points": [],
        "reply_tone": None,
        "reply_policy": [],
    },
    {
        "id": "e003",
        "from": "alice.wang@company.com",
        "to": "support@company.com",
        "subject": "Invoice #INV-2024-0892 - Payment Confirmation Request",
        "body": (
            "Hello,\n\nI am following up on invoice #INV-2024-0892 for $4,750 submitted on January 8th. "
            "Our payment terms are Net-15 and the due date is approaching. "
            "Could you confirm receipt and expected payment date?\n\n"
            "Please find the invoice attached.\n\nBest regards,\nAlice Wang\nAccounts Receivable"
        ),
        "timestamp": "2024-01-15T10:15:00Z",
        "has_attachment": True,
        "correct_category": "billing",
        "correct_priority": "medium",
        "correct_department": "finance",
        "requires_reply": True,
        "reply_key_points": [
            "acknowledge receipt of invoice",
            "confirm processing timeline",
            "provide contact for follow-up",
        ],
        "reply_tone": "professional",
        "reply_policy": ["do not commit to specific payment date without finance approval"],
    },
    {
        "id": "e004",
        "from": "tom.chen@company.com",
        "to": "support@company.com",
        "subject": "Feature Request: Dark mode for dashboard",
        "body": (
            "Hi,\n\nLove the product! One thing I'd really like to see is a dark mode option "
            "for the analytics dashboard. Many of us work late and the bright white interface "
            "is quite harsh on the eyes. I know several colleagues who've mentioned the same thing.\n\n"
            "Would this be on the roadmap?\n\nThanks,\nTom"
        ),
        "timestamp": "2024-01-15T11:30:00Z",
        "has_attachment": False,
        "correct_category": "feature_request",
        "correct_priority": "low",
        "correct_department": "product",
        "requires_reply": True,
        "reply_key_points": [
            "thank customer for feedback",
            "acknowledge the request",
            "explain feedback process or roadmap visibility",
        ],
        "reply_tone": "friendly",
        "reply_policy": ["do not promise feature delivery dates"],
    },
    {
        "id": "e005",
        "from": "legal@enterprise-partner.com",
        "to": "support@company.com",
        "subject": "Data Processing Agreement — Action Required by Jan 20",
        "body": (
            "Dear Team,\n\nAs per our ongoing GDPR compliance review, we require an updated "
            "Data Processing Agreement (DPA) signed by your organization before January 20, 2024. "
            "Failure to provide this will require us to suspend data sharing under our current contract.\n\n"
            "Please route this to your Legal or Compliance team immediately.\n\n"
            "Attached: DPA_Template_v3.pdf\n\nRegards,\nLegal Department\nEnterprise Partner LLC"
        ),
        "timestamp": "2024-01-15T07:20:00Z",
        "has_attachment": True,
        "correct_category": "legal",
        "correct_priority": "high",
        "correct_department": "legal",
        "requires_reply": True,
        "reply_key_points": [
            "acknowledge receipt",
            "confirm routing to legal team",
            "provide expected response timeline",
        ],
        "reply_tone": "professional",
        "reply_policy": ["do not make legal commitments", "route to legal immediately"],
    },
    {
        "id": "e006",
        "from": "hr@company.com",
        "to": "allstaff@company.com",
        "subject": "Reminder: Submit your Q4 timesheet by Friday",
        "body": (
            "Hi All,\n\nThis is a reminder that Q4 timesheets are due this Friday by 5pm. "
            "Please log in to the HR portal and submit your hours.\n\n"
            "Contact hr@company.com with any questions.\n\nThanks,\nHR Team"
        ),
        "timestamp": "2024-01-15T09:00:00Z",
        "has_attachment": False,
        "correct_category": "internal",
        "correct_priority": "low",
        "correct_department": "hr",
        "requires_reply": False,
        "reply_key_points": [],
        "reply_tone": None,
        "reply_policy": [],
    },
    {
        "id": "e007",
        "from": "mark.roberts@bigcorp.com",
        "to": "sales@company.com",
        "subject": "Interested in Enterprise Plan — 500 seat license",
        "body": (
            "Hello,\n\nWe are evaluating your platform for our enterprise team of approximately 500 users. "
            "We've completed a 30-day trial with a small group and results have been very positive.\n\n"
            "We'd like to discuss volume pricing, SSO integration, and dedicated support SLAs. "
            "Who should I speak with to move this forward?\n\nBest,\nMark Roberts\nDirector of IT, BigCorp"
        ),
        "timestamp": "2024-01-15T13:00:00Z",
        "has_attachment": False,
        "correct_category": "sales",
        "correct_priority": "high",
        "correct_department": "sales",
        "requires_reply": True,
        "reply_key_points": [
            "express enthusiasm",
            "thank for positive trial results",
            "offer to schedule a call",
            "mention account executive or sales contact",
        ],
        "reply_tone": "enthusiastic",
        "reply_policy": ["do not quote pricing without AE involvement"],
    },
    {
        "id": "e008",
        "from": "security-alerts@company.com",
        "to": "support@company.com",
        "subject": "[AUTOMATED] Failed login attempts detected — Account user_4821",
        "body": (
            "AUTOMATED SECURITY ALERT\n\n"
            "Account: user_4821 (jane.doe@client.com)\n"
            "Event: 15 failed login attempts in 10 minutes\n"
            "Source IPs: 185.220.101.x (Tor exit node)\n"
            "Time: 2024-01-15 08:55 UTC\n\n"
            "Recommended action: Review and consider temporary account lock.\n"
            "This is an automated message from the security monitoring system."
        ),
        "timestamp": "2024-01-15T08:57:00Z",
        "has_attachment": False,
        "correct_category": "security",
        "correct_priority": "critical",
        "correct_department": "security",
        "requires_reply": False,
        "reply_key_points": [],
        "reply_tone": None,
        "reply_policy": [],
    },
    {
        "id": "e009",
        "from": "priya.sharma@startup.io",
        "to": "support@company.com",
        "subject": "How do I export data to CSV?",
        "body": (
            "Hi support,\n\nI've been using your product for a few weeks and love it! "
            "I can't figure out how to export my project data to CSV format. "
            "I've looked through the docs but can't find it.\n\n"
            "Thanks!\nPriya"
        ),
        "timestamp": "2024-01-15T14:00:00Z",
        "has_attachment": False,
        "correct_category": "support",
        "correct_priority": "medium",
        "correct_department": "support",
        "requires_reply": True,
        "reply_key_points": [
            "answer the question with clear steps",
            "be friendly and concise",
            "offer further help",
        ],
        "reply_tone": "friendly",
        "reply_policy": ["provide accurate instructions"],
    },
    {
        "id": "e010",
        "from": "ceo@company.com",
        "to": "support@company.com",
        "subject": "FWD: Major client complaint - needs attention",
        "body": (
            "Team — please handle this immediately. This is our second-largest account.\n\n"
            "--- Forwarded message ---\n"
            "From: director@enterprise-client.com\n"
            "Subject: Extremely disappointed with recent service\n\n"
            "We have experienced three separate outages this month and have received no "
            "proactive communication from your team. We are reconsidering our contract renewal "
            "which is due in 60 days. I expect a call from your VP of Customer Success today.\n\n"
            "Director of Operations\nEnterprise Client Inc."
        ),
        "timestamp": "2024-01-15T15:30:00Z",
        "has_attachment": False,
        "correct_category": "escalation",
        "correct_priority": "critical",
        "correct_department": "customer_success",
        "requires_reply": True,
        "reply_key_points": [
            "sincerely apologize",
            "acknowledge specific failures",
            "commit to VP of Customer Success outreach",
            "mention retention risk awareness",
        ],
        "reply_tone": "professional",
        "reply_policy": ["acknowledge failures honestly", "do not make promises on uptime without engineering input"],
    },
]

CATEGORIES = ["support", "spam", "billing", "feature_request", "legal", "internal", "sales", "security", "escalation"]
PRIORITIES = ["critical", "high", "medium", "low"]
DEPARTMENTS = ["engineering", "finance", "product", "legal", "hr", "sales", "support", "security", "customer_success", None]
