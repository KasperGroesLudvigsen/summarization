import pandas as pd

data = {"text": [
"""Beboerne i Marielyst kan måske se frem til et 3,5 etager højt hotel eller ferieboliger.

Det er i hvert fald udsigterne med den nye lokalplan, der i går aftes blev godkendt i Guldborgsund Kommunes teknik-, klima- og miljøudvalg.

Lokalplanen indeholder retningslinjer for skilte, facader og byggeri.

Målet med planen er at ensrette retningslinjerne på Marielyst Strandgade og at gøre det mere attraktivt at være erhvervsdrivende i byen, siger udvalgsformand Jesper Blomberg fra Moderaterne.

- Vi ønsker, at Marielyst skal blive endnu mere attraktiv at besøge hele året. Derfor er lokalplanen ambitiøs og fremtidssikret i forhold til at give mulighed for nye aktiviteter, siger han.

Sagen skal nu videre til byrådet, hvorefter det forventes, at planen bliver sendt i offentlig høring fra januar til marts 2025.""",
"""Forældrene til 35 børn på Samsø har i dag været kreative, når det kommer til at få passet deres børn.

Den private børnehave, Børnehuset på Samsø, skal nemlig tages under konkursbehandling, og kan derfor fra i dag ikke længere passe børn.

Det betyder, at forældre har måtte finde en midlertidig løsning, hvor de, der har haft mulighed for det, selv er trådt til og passer børnene i egne hjem.

Samsø Børnehus er én ud af to børnehaver på øen. Den anden er den kommunale børnehave Rumplepotten, som har en gennemsnitsnormering på cirka 45 børn.

Forvaltningschef for Børn, Unge og Kultur i Samsø Kommune, Helle Griebel Anesen, skriver til DR i en SMS, at kommunen ikke kan udtale sig, før konkursen er begæret."""
], 
"summary_reference": [
    "Kendt turistområde skal have lov til at opføre 3,5 etager højt byggeri",
    "Omkring 35 børn på Samsø mangler akut pasningstilbud"
]}

data = pd.DataFrame.from_records(data)
data.to_csv("mock_evaluation_data.csv", index=False)