"""
Fake News Detection Using NLP - Synthetic News Dataset Generator
Author: Pragna | Professional NLP Engineer & Data Science Intern
Description: Generates a balanced, linguistically distinct fake vs real news dataset.
             Developed by Pragna.
"""

import os
import random
import numpy as np
import pandas as pd

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

def generate_synthetic_news_data(num_records=1000):
    print(f"Generating {num_records} balanced fake and real news articles...")
    
    # Vocabulary templates for FAKE news
    fake_subjects = [
        "Secret government insiders", "Alien spaceship wreckage", "Shocking leaked documents",
        "A clandestine medical cover-up", "Undeniable evidence", "The global elite",
        "A shadow organization", "An anonymous whistleblower", "Shocking scientific proof"
    ]
    fake_verbs = [
        "exposed a shocking conspiracy regarding", "revealed the unbelievable truth about",
        "uncovered a classified agenda concerning", "admitted to a secret operation involving",
        "exposed the undeniable reality of", "confirms the secret truth behind"
    ]
    fake_objects = [
        "microchip implants in daily vaccines.", "a fake moon landing staging site.",
        "weather modification machines controlled by shadow groups.", "a cover-up of extraterrestrial encounters.",
        "a plan to replace fiat currency with secret mind-control chips.",
        "fake water chemicals designed to control public thoughts."
    ]
    fake_clickbaits = [
        "SHOCKING TRUTH!", "MUST READ!", "THEY DON'T WANT YOU TO KNOW!", "UNBELIEVABLE!",
        "THE TRUTH EXPOSED!", "ACTUAL PROOF!"
    ]
    fake_text_bodies = [
        "This is a shocking revelation that they don't want you to know. A highly placed secret government insider has leaked classified documents confirming that a shadow organization is actively running weather control machines. Independent researchers have found undeniable evidence, but mainstream media refuses to report this conspiracy. Share this before it gets taken down! It's time to wake up!",
        "Unbelievable truth exposed! An anonymous whistleblower has finally admitted that a clandestine medical cover-up was staged by the global elite. Sources say secret microchip implants have been hidden inside daily vaccines for years to control thought processes. If you look at the raw data, the proof is absolutely undeniable. Do your own research!",
        "Breaking news: actual proof has surfaced about alien spaceship wreckage discovered under deep ocean chambers. A secret military branch has been keeping this classified technology hidden from the public to advance their secret mind-control agenda. Experts who tried to expose this shocking secret have vanished. Read the full leak here!",
        "The global elite are hiding weather modification machines that cause unnatural storms. A shocking leak reveals that a secret group is pulling the strings. This is a must-read warning! The undeniable reality is that our thoughts are being modified by fake chemicals in public drinking water. The truth is finally coming to light!"
    ]

    # Vocabulary templates for REAL news
    real_subjects = [
        "The Department of Finance", "Parliamentary representatives", "A team of clinical researchers",
        "The Prime Minister's press secretary", "The central banking authorities", "The World Health organization",
        "Official government agencies", "The international trade council", "State environmental scientists"
    ]
    real_verbs = [
        "announced the implementation of a new policy on", "published a comprehensive report concerning",
        "approved the proposed reform legislation targeting", "confirmed a strategic agreement regarding",
        "issued an official statement clarifying details of", "revealed research results on"
    ]
    real_objects = [
        "small business tax regulations.", "sustainable energy development funding.",
        "public health safety protocols.", "bilateral trade tariffs on imported goods.",
        "academic curriculum standards in rural schools.", "infrastructure investments in public transport networks."
    ]
    real_text_bodies = [
        "The Department of Finance issued an official statement yesterday confirming the implementation of a new policy designed to support small business growth. According to parliamentary spokespeople, the reform legislation introduces tax reductions for local entrepreneurs. Financial analysts state that the infrastructure funding will stimulate regional economic activity and increase employment indices over the fiscal year.",
        "In a comprehensive report published this morning, a team of clinical researchers confirmed significant progress in sustainable energy initiatives. The study, funded by state environmental agencies, indicates that solar energy grid integration has improved by fifteen percent over the last quarter. State representatives announced that additional public funding will be allocated to expand municipal transport networks.",
        "Official representatives of the international trade council approved the proposed bilateral trade tariffs during the summit. The official agreement aims to stabilize commodity pricing and support domestic manufacturing sectors. Government officials declared that regular audits will be performed to ensure compliance with international standard regulations.",
        "The World Health Organization published a comprehensive study highlighting modern public health safety protocols. The report outlines guidelines designed to prevent seasonal viral transmission in metropolitan areas. Medical professionals emphasized that systematic community education and vaccine distribution remain the most effective methods for long-term health stabilization."
    ]
    
    titles = []
    texts = []
    labels = []
    
    # Generate 500 Fake news articles
    for i in range(num_records // 2):
        clickbait = random.choice(fake_clickbaits)
        subj = random.choice(fake_subjects)
        verb = random.choice(fake_verbs)
        obj = random.choice(fake_objects)
        title = f"{clickbait} {subj} {verb} {obj}"
        
        # Build slightly randomized text body
        base_text = random.choice(fake_text_bodies)
        extra_noise = f" This secret conspiracy involves {random.choice(fake_subjects).lower()} and has been running for decades. Shocking truth revealed!"
        text = base_text + extra_noise
        
        titles.append(title)
        texts.append(text)
        labels.append("FAKE")
        
    # Generate 500 Real news articles
    for i in range(num_records // 2):
        subj = random.choice(real_subjects)
        verb = random.choice(real_verbs)
        obj = random.choice(real_objects)
        title = f"{subj} {verb} {obj}"
        
        # Build slightly randomized text body
        base_text = random.choice(real_text_bodies)
        extra_info = f" According to the published statistics, {random.choice(real_subjects).lower()} will oversee the regulatory audits to maintain public standards."
        text = base_text + extra_info
        
        titles.append(title)
        texts.append(text)
        labels.append("REAL")
        
    # Combine into DataFrame
    df = pd.DataFrame({
        "title": titles,
        "text": texts,
        "label": labels
    })
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    # Create data folder
    os.makedirs("D:\\GITHUB\\Fake-News-Detection\\data", exist_ok=True)
    
    df = generate_synthetic_news_data(1000)
    
    output_path = "D:\\GITHUB\\Fake-News-Detection\\data\\fake_or_real_news.csv"
    df.to_csv(output_path, index=False)
    print(f"\nDataset successfully created and saved to: {output_path}")
    print("\nDataset Info:")
    print(df.info())
    print("\nClass Distribution:")
    print(df["label"].value_counts())
