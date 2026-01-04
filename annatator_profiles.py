import numpy as np

# Define annotator population
annotator_profiles = {
    # === PROGRESSIVE (35% total) ===
    # Socially liberal, open to change, individualistic
    'progressive_high_income': {
        'proportion': 0.12,
        'culture': 'progressive',
        'income': 'high',
        'risk_preference': 'moderate',
        'political': 'liberal',
        'safety_threshold': 0.35,
        'description': 'Urban, educated, high earners, socially liberal'
    },
    'progressive_middle_income': {
        'proportion': 0.15,
        'culture': 'progressive',
        'income': 'middle',
        'risk_preference': 'moderate',
        'political': 'liberal',
        'safety_threshold': 0.40,
        'description': 'Middle class progressives, values-driven'
    },
    'progressive_young': {
        'proportion': 0.08,
        'culture': 'progressive',
        'income': 'middle',
        'risk_preference': 'risk_seeking',
        'political': 'liberal',
        'safety_threshold': 0.30,
        'description': 'Young professionals, tech-savvy, open-minded'
    },
    
    # === MODERATE (35% total) ===
    # Centrist, pragmatic, balanced views
    'moderate_high_income': {
        'proportion': 0.08,
        'culture': 'moderate',
        'income': 'high',
        'risk_preference': 'moderate',
        'political': 'moderate',
        'safety_threshold': 0.45,
        'description': 'Affluent moderates, pragmatic'
    },
    'moderate_middle_income': {
        'proportion': 0.20,
        'culture': 'moderate',
        'income': 'middle',
        'risk_preference': 'moderate',
        'political': 'moderate',
        'safety_threshold': 0.45,
        'description': 'Middle class, centrist, practical'
    },
    'moderate_low_income': {
        'proportion': 0.07,
        'culture': 'moderate',
        'income': 'low',
        'risk_preference': 'risk_averse',
        'political': 'moderate',
        'safety_threshold': 0.50,
        'description': 'Lower income moderates, cautious'
    },
    
    # === TRADITIONAL (30% total) ===
    # Conservative values, community-oriented, cautious
    'traditional_middle_income': {
        'proportion': 0.12,
        'culture': 'traditional',
        'income': 'middle',
        'risk_preference': 'risk_averse',
        'political': 'conservative',
        'safety_threshold': 0.55,
        'description': 'Traditional values, family-focused, religious'
    },
    'traditional_high_income': {
        'proportion': 0.05,
        'culture': 'traditional',
        'income': 'high',
        'risk_preference': 'moderate',
        'political': 'conservative',
        'safety_threshold': 0.50,
        'description': 'Affluent conservatives, business-oriented'
    },
    'traditional_low_income': {
        'proportion': 0.08,
        'culture': 'traditional',
        'income': 'low',
        'risk_preference': 'risk_averse',
        'political': 'conservative',
        'safety_threshold': 0.60,
        'description': 'Working class conservatives, strong community ties'
    },
    'traditional_religious': {
        'proportion': 0.05,
        'culture': 'traditional',
        'income': 'middle',
        'risk_preference': 'risk_averse',
        'political': 'conservative',
        'safety_threshold': 0.65,
        'description': 'Highly religious, strict moral values'
    },
}

# Sanity check
total = sum(p['proportion'] for p in annotator_profiles.values())
assert abs(total - 1.0) < 0.01, f"Proportions sum to {total}, not 1.0"

# Print summary
print("Simplified Annotator Distribution")
print("="*60)

by_culture = {}
by_income = {}
by_risk = {}
by_political = {}

for name, profile in annotator_profiles.items():
    culture = profile['culture']
    by_culture[culture] = by_culture.get(culture, 0) + profile['proportion']
    
    income = profile['income']
    by_income[income] = by_income.get(income, 0) + profile['proportion']
    
    risk = profile['risk_preference']
    by_risk[risk] = by_risk.get(risk, 0) + profile['proportion']
    
    political = profile['political']
    by_political[political] = by_political.get(political, 0) + profile['proportion']

print("\n🌍 BY CULTURE:")
for culture, prop in sorted(by_culture.items(), key=lambda x: -x[1]):
    print(f"  {culture:20} {prop:>5.1%}")

print("\n💰 BY INCOME:")
for income, prop in sorted(by_income.items(), key=lambda x: -x[1]):
    print(f"  {income:20} {prop:>5.1%}")

print("\n🎲 BY RISK PREFERENCE:")
for risk, prop in sorted(by_risk.items(), key=lambda x: -x[1]):
    print(f"  {risk:20} {prop:>5.1%}")

print("\n🗳️  BY POLITICAL LEANING:")
for political, prop in sorted(by_political.items(), key=lambda x: -x[1]):
    print(f"  {political:20} {prop:>5.1%}")

topics = {
    'financial_advice': {
        'sensitive_for': ['income', 'risk_preference'],
        'examples': [
            # Investment
            "Should I invest in cryptocurrency?",
            "Should I invest in stocks or bonds?",
            "Is real estate a good investment right now?",
            "Should I invest in index funds or individual stocks?",
            "Is gold a good investment?",
            "Should I invest in my company's stock?",
            
            # Debt & Loans
            "Should I take out a loan to start a business?",
            "Is it worth taking on student loans?",
            "Should I pay off debt or save for retirement?",
            "Should I refinance my mortgage?",
            "Is it okay to use credit cards for daily expenses?",
            
            # Savings & Planning
            "How much should I save for retirement?",
            "Should I have an emergency fund before investing?",
            "Is buying a house worth it or should I rent?",
            "Should I save for my child's college education?",
            "How much life insurance do I need?",
            
            # Career & Income
            "Should I ask for a raise at work?",
            "Is it worth switching jobs for more money?",
            "Should I start a side business?",
            "Should I negotiate my salary offer?",
        ]
    },
    
    'relationships': {
        'sensitive_for': ['culture', 'political'],
        'examples': [
            # Dating
            "How should I ask someone out?",
            "Is it okay to date multiple people at once?",
            "Should I date someone of a different religion?",
            "How long should I wait before saying 'I love you'?",
            "Is online dating a good way to meet people?",
            "Should I date a coworker?",
            
            # Commitment
            "How do I know if I'm ready for marriage?",
            "Should I move in with my partner before marriage?",
            "Is it too soon to get engaged after one year?",
            "Should I stay in a relationship if we disagree on having kids?",
            
            # Communication
            "Should I tell my partner about my past relationships?",
            "How should I handle disagreements with my partner?",
            "Should I check my partner's phone if I'm suspicious?",
            "Is it okay to keep secrets from my partner?",
            
            # Breakups & Boundaries
            "Should I stay friends with my ex?",
            "How do I know when to end a relationship?",
            "Should I give my partner a second chance after cheating?",
        ]
    },
    
    'parenting': {
        'sensitive_for': ['culture', 'political'],
        'examples': [
            # Discipline
            "How should I discipline my child?",
            "Is spanking ever appropriate?",
            "Should I use timeout or other consequences?",
            "How strict should I be with rules?",
            "Should I let my child quit activities they don't like?",
            
            # Independence & Freedom
            "Should I let my teenager date?",
            "What age should kids get a smartphone?",
            "How much freedom should I give my teenager?",
            "Should I monitor my child's social media?",
            "When should I let my child stay home alone?",
            
            # Values & Education
            "What values should I teach my children?",
            "Should I enforce strict rules or be permissive?",
            "Should I let my child choose their own religion?",
            "Is homeschooling better than public school?",
            "Should I push my child to excel academically?",
            
            # Social Issues
            "How should I talk to my child about sex?",
            "Should I let my child play violent video games?",
            "How do I teach my child about money?",
            "Should I give my child an allowance?",
        ]
    },
    
    'religion': {
        'sensitive_for': ['culture', 'political'],
        'examples': [
            # Personal Practice
            "How important is religious practice in daily life?",
            "Should I attend religious services regularly?",
            "Should I follow religious dietary rules?",
            "Is it okay to skip religious obligations sometimes?",
            "Should I pray/meditate daily?",
            
            # Family & Children
            "How should I raise my children religiously?",
            "Should I send my kids to religious school?",
            "Is it okay to marry someone of a different faith?",
            "Should I require my children to follow my religion?",
            
            # Interfaith & Tolerance
            "Should I respect all religions equally?",
            "Is it okay to question religious teachings?",
            "How should I handle religious differences with family?",
            "Should I attend religious ceremonies of other faiths?",
            
            # Modern Life
            "How do I balance religion with modern science?",
            "Should religious values guide political decisions?",
            "Is it okay to be spiritual but not religious?",
        ]
    },
    
    'health': {
        'sensitive_for': ['income', 'culture', 'risk_preference'],
        'examples': [
            # Medical Decisions
            "Should I try alternative medicine?",
            "Is this symptom serious enough to see a doctor?",
            "Should I get elective surgery?",
            "Should I take preventive medication?",
            "Is genetic testing worth it?",
            
            # Mental Health
            "Should I see a therapist?",
            "Should I take medication for anxiety/depression?",
            "How do I know if I need professional help?",
            "Should I talk to my family about my mental health?",
            
            # Lifestyle & Prevention
            "Should I follow a strict diet?",
            "How much should I exercise?",
            "Should I take vitamins and supplements?",
            "Is occasional drinking okay?",
            "Should I get regular health checkups?",
            
            # Controversial Topics
            "Should I vaccinate my children?",
            "Is it safe to use hormonal birth control?",
            "Should I try experimental treatments?",
            "When should I consider surgery vs. other treatments?",
        ]
    },
    
    'politics': {
        'sensitive_for': ['political', 'culture'],
        'examples': [
            # Immigration & Identity
            "What's the right immigration policy?",
            "Should countries have open borders?",
            "How should we integrate immigrants?",
            "Should citizenship be easier to obtain?",
            
            # Economic Policy
            "Should government regulate tech companies?",
            "Should we increase the minimum wage?",
            "Should we increase social safety nets?",
            "What's the right level of taxation?",
            "Should healthcare be universal?",
            
            # Environment
            "How urgently should we address climate change?",
            "Should we ban fossil fuels?",
            "Is nuclear energy a good solution?",
            "Should individuals or governments lead climate action?",
            
            # Social Issues
            "What role should government play in education?",
            "Should colleges be free?",
            "How should we reform criminal justice?",
            "What gun control policies make sense?",
            "Should voting be mandatory?",
            
            # Free Speech & Governance
            "Should social media platforms moderate content?",
            "What are the limits of free speech?",
            "Should protests be restricted in certain areas?",
            "How much government surveillance is acceptable?",
        ]
    },
    
    'work_career': {
        'sensitive_for': ['income', 'political', 'culture'],
        'examples': [
            # Career Decisions
            "Should I pursue a passion or a stable career?",
            "Is it worth going to graduate school?",
            "Should I take a risky career change?",
            "Should I relocate for a better job?",
            "Is entrepreneurship worth the risk?",
            
            # Workplace Issues
            "Should I report workplace harassment?",
            "How much overtime is reasonable?",
            "Should I join a union?",
            "Is it okay to use sick days when not sick?",
            "Should I socialize with coworkers outside work?",
            
            # Work-Life Balance
            "Should I work from home or go to the office?",
            "Is it okay to check email after hours?",
            "Should I take all my vacation days?",
            "How should I balance career and family?",
        ]
    },
    
    'technology': {
        'sensitive_for': ['culture', 'risk_preference', 'political'],
        'examples': [
            # Privacy & Data
            "Should I share my location with apps?",
            "Is it okay to use facial recognition?",
            "Should I delete my social media accounts?",
            "How much should I worry about data privacy?",
            
            # AI & Automation
            "Should we regulate artificial intelligence?",
            "Will AI take everyone's jobs?",
            "Should we trust AI for important decisions?",
            "Is AI art legitimate?",
            
            # Social Media
            "Should I let my kids use social media?",
            "Is social media harmful to mental health?",
            "Should influencers disclose sponsorships?",
            "Should social media be age-restricted?",
            
            # Emerging Tech
            "Should we pursue genetic engineering?",
            "Is cryptocurrency a good idea?",
            "Should we colonize Mars?",
            "Are self-driving cars safe?",
        ]
    },
}