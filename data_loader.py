import pandas as pd
import io

def load_data(file):
    """
    Charge les données à partir d'un fichier CSV, Excel ou texte brut
    et retourne un DataFrame.
    Le fichier doit contenir une colonne "review".
    """
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file)
        elif file.name.endswith(".txt"):
            content = file.getvalue().decode("utf-8").splitlines()
            df = pd.DataFrame(content, columns=["review"])
        else:
            return None

        if "review" not in df.columns:
            raise ValueError("Le fichier doit contenir une colonne 'review'.")
        return df.dropna(subset=["review"])
    except Exception as e:
        print(f"Erreur lors du chargement du fichier : {e}")
        return None
