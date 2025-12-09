import streamlit as st
import lancedb

st.title("LanceDB Viewer")

# Use the correct path
db = lancedb.connect("./app/db/ghostwriter_db")

# List available tables
table_names = db.table_names()
st.write("Available tables:", table_names)

# Let user select table or use default
if table_names:
    selected_table = st.selectbox("Select table", table_names, index=0)
    table = db.open_table(selected_table)
    
    df = table.to_pandas()
    st.write(f"Table: {selected_table}")
    st.write(f"Rows: {len(df)}")
    st.dataframe(df)
else:
    st.warning("No tables found in the database.")