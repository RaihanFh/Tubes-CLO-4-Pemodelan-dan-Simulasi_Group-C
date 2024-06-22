import streamlit as st

# Title and group members
st.title("Prediksi Harga Saham dengan Kombinasi Machine Learning dan Gerakan Brown Geometrik (GBM)")
st.markdown("""
### Anggota Kelompok:
- **Adhie Haqqi Ramadhani S.** - 1301213312
- **Naufal Alfarisi** - 1301213452
- **Mufidah Alfiah** - 1301184180
- **Raihan Fadhilah Hafiizh** - 1301213113
""")

# Load the uploaded file
uploaded_file = st.file_uploader("Upload your Jupyter Notebook (.ipynb) file", type=["ipynb"])

if uploaded_file is not None:
    import nbformat
    from nbconvert import MarkdownExporter
    from io import StringIO

    # Read the uploaded Jupyter notebook
    notebook_content = uploaded_file.getvalue()
    notebook_node = nbformat.reads(notebook_content, as_version=4)

    # Convert the notebook to markdown
    exporter = MarkdownExporter()
    markdown, _ = exporter.from_notebook_node(notebook_node)
    st.markdown(markdown, unsafe_allow_html=True)
else:
    st.warning("Please upload a Jupyter Notebook file.")

# Additional Streamlit components can be added here for the project
