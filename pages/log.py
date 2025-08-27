import streamlit as st
import pandas as pd
from datetime import datetime

# Logdaten werden aus st.session_state["visit_log"] geladen und gespeichert

if "visit_log" not in st.session_state:
    st.session_state["visit_log"] = pd.DataFrame(columns=["ID", "Zeit", "Eintrag"])

def load_logs():
    return st.session_state["visit_log"]

def save_logs(df):
    st.session_state["visit_log"] = df

st.title("Log-Einträge bearbeiten")

logs = load_logs()

# zwei spalten: spalte 1 ändern, spalte 2 löschen und hinzufügen
col_left, col_right = st.columns(2)


with col_left:

    if not logs:
        st.info("Noch keine Log-Einträge vorhanden.")
    else:
        # logs ist eine Liste
        selected_index_edit = st.selectbox("Wählen Sie einen Log-Eintrag zum Bearbeiten aus:", range(len(logs)))
        selected_row = logs[selected_index_edit]
        # Zeit als datetime parsen
        try:
            current_time = pd.to_datetime(selected_row[1])
        except Exception:
            current_time = datetime.now()

        new_date = st.date_input("Neues Datum", value=current_time.date())
        new_time = st.text_input("Neue Zeit (HH:MM:SS)", value=current_time.strftime("%H:%M:%S"))

        if st.button("Zeit aktualisieren"):
            # Datum beibehalten, nur Zeit ändern
            new_datetime = datetime.combine(new_date, datetime.strptime(new_time, "%H:%M:%S").time())
            st.write(f"Neue Zeit: {new_datetime}")
            logs[selected_index_edit] = [selected_row[0], new_datetime.strftime("%Y-%m-%d %H:%M:%S")]
            save_logs(logs)
            st.success("Zeit wurde aktualisiert.")

with col_right:

    pass

    # selected_index_del = st.selectbox("Wählen Sie einen Log-Eintrag zum Löschen aus:", range(len(logs)))
    # if st.button("Log-Eintrag löschen"):
    #     logs.pop(selected_index_del)
    #     save_logs(logs)
    #     st.success("Log-Eintrag wurde gelöscht.")

    # selected_index_add = st.selectbox("Hinter welchem Log-Eintrag möchten Sie hinzufügen?", range(len(logs)))
    # # selectbox mit allen möglichen bojen die mit der ausgewählten verbunden sind
    # new_buoy = st.selectbox("Boje", st.session_state["selected_buoys"])
    # new_buoy_date = st.date_input("Datum", value=datetime.now())
    # new_buoy_time = st.text_input("Zeit (HH:MM)", value=datetime.now().strftime("%H:%M"))
    # if st.button("Log-Eintrag hinzufügen"):
    #     new_log = [new_buoy, datetime.combine(new_buoy_date, datetime.strptime(new_buoy_time, "%H:%M").time()).strftime("%Y-%m-%d %H:%M")]
    #     logs.append(new_log)
    #     save_logs(logs)
    #     st.success("Neuer Log-Eintrag wurde hinzugefügt.")

# check if the order of logs are valid. log[i][1] < log[i+1][1]
for i in range(len(logs) - 1):
    if logs[i][1] >= logs[i + 1][1]:
        st.warning(f"Die Zeitangabe für Log-Eintrag {i} ist nicht vor der Zeitangabe für Log-Eintrag {i + 1}.")

st.write("Aktuelle Log-Einträge:")

# Dataframe aus Einträgen von logs.
log_df = pd.DataFrame(logs, columns=["Boje", "Zeit"])
st.dataframe(log_df)
