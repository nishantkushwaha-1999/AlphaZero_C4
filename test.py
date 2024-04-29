import streamlit as st

def style_button_row(clicked_button_ix, n_buttons):
    def get_button_indices(button_ix):
        return {
            'nth_child': button_ix,
            'nth_last_child': n_buttons - button_ix + 1
        }

    clicked_style = """
    div[data-testid*="stHorizontalBlock"] > div:nth-child(%(nth_child)s):nth-last-child(%(nth_last_child)s) button {
        border-color: rgb(255, 75, 75);
        color: rgb(255, 75, 75);
        box-shadow: rgba(255, 75, 75, 0.5) 0px 0px 0px 0.2rem;
        outline: currentcolor none medium;
    }
    """
    unclicked_style = """
    div[data-testid*="stHorizontalBlock"] > div:nth-child(%(nth_child)s):nth-last-child(%(nth_last_child)s) button {
        pointer-events: none;
        cursor: not-allowed;
        opacity: 0.65;
        filter: alpha(opacity=65);
        -webkit-box-shadow: none;
        box-shadow: none;
    }
    """
    style = ""
    for ix in range(n_buttons):
        ix += 1
        if ix == clicked_button_ix:
            style += clicked_style % get_button_indices(ix)
        else:
            style += unclicked_style % get_button_indices(ix)
    st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.sidebar.columns([1, 1, 1, 1])

with col1:
    st.button("📆", on_click=style_button_row, kwargs={
        'clicked_button_ix': 1, 'n_buttons': 4
    })
with col2:
    st.button("👌", on_click=style_button_row, kwargs={
        'clicked_button_ix': 2, 'n_buttons': 4
    })
with col3:
    st.button("◀", on_click=style_button_row, kwargs={
       'clicked_button_ix': 3, 'n_buttons': 4

    })
with col4:
    st.button("🚧", on_click=style_button_row, kwargs={
        'clicked_button_ix': 4, 'n_buttons': 4
    })
