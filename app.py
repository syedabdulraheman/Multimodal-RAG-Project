"""
Streamlit Frontend for Multimodal RAG System
File: app.py

Run with: streamlit run app.py
"""

import streamlit as st
import os
import tempfile
import sys
import io

# Import from multimodal_rag.py
from multimodal_rag import (
    process_pdf,
    MultimodalVectorStore,
    embed_text,
    answer_with_image,
    generate_text_answer,
    clip_model,
    blip_model,
    text_model
)

# Page configuration
st.set_page_config(
    page_title="Multimodal RAG Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #145a8c;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .answer-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        color: #000000;
    }
    .answer-box p {
        color: #000000;
        margin: 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'image_store' not in st.session_state:
    st.session_state.image_store = None
if 'documents' not in st.session_state:
    st.session_state.documents = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = None

# Header
st.markdown('<h1 class="main-header">📄 Multimodal RAG Q&A System</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem;'>Upload a PDF with text and images, then ask questions about it!</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings & Info")
    
    # Model status
    st.subheader("🤖 Model Status")
    with st.spinner("Checking models..."):
        if clip_model and blip_model and text_model:
            st.success("✅ All models loaded")
            st.caption("CLIP, BLIP, and FLAN-T5 ready")
        else:
            st.error("❌ Models not loaded")
    
    st.divider()
    
    # PDF Status
    if st.session_state.pdf_processed:
        st.subheader("📄 Current PDF")
        st.info(f"**{st.session_state.pdf_name}**")
        text_count = sum(1 for d in st.session_state.documents if d['type'] == 'text')
        image_count = sum(1 for d in st.session_state.documents if d['type'] == 'image')
        st.metric("Text Chunks", text_count)
        st.metric("Images", image_count)
        st.metric("Total Documents", len(st.session_state.documents))
    
    st.divider()
    
    # Advanced settings
    st.subheader("🔧 Advanced Settings")
    k_results = st.slider(
        "Results to retrieve",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of relevant documents to retrieve for each query"
    )
    
    show_details = st.checkbox(
        "Show retrieval details",
        value=True,
        help="Display which documents were retrieved"
    )
    
    st.divider()
    
    # Information
    st.subheader("ℹ️ About")
    st.info(
        """
        **This system uses:**
        - 🎯 CLIP for embeddings
        - 🖼️ BLIP for image understanding
        - 📝 FLAN-T5 for answer generation
        
        **Features:**
        - ✅ 100% Local processing
        - ✅ No API keys needed
        - ✅ Handles text + images
        - ✅ Privacy-focused
        """
    )
    
    # Reset button
    st.divider()
    if st.button("🔄 Reset Session", type="secondary", use_container_width=True):
        st.session_state.vector_store = None
        st.session_state.image_store = None
        st.session_state.documents = None
        st.session_state.chat_history = []
        st.session_state.pdf_processed = False
        st.session_state.pdf_name = None
        st.success("Session reset!")
        st.rerun()

# Main content area - Two columns
col1, col2 = st.columns([1, 1], gap="large")

# LEFT COLUMN - Upload PDF
with col1:
    st.header("📤 Step 1: Upload PDF")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF containing text and/or images",
        key="pdf_uploader"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File selected: **{uploaded_file.name}**")
        
        # Show file info
        file_size = uploaded_file.size / 1024  # Convert to KB
        st.caption(f"File size: {file_size:.2f} KB")
        
        # Process button
        if st.button("🔍 Process PDF", type="primary", use_container_width=True):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # Progress bar and status
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Process PDF
                status_text.text("📖 Reading PDF...")
                progress_bar.progress(25)
                documents, embeddings, image_store = process_pdf(tmp_path)
                
                # Step 2: Create vector store
                status_text.text("🔗 Creating vector store...")
                progress_bar.progress(50)
                vector_store = MultimodalVectorStore(embeddings, documents)
                
                # Step 3: Store in session
                status_text.text("💾 Saving to session...")
                progress_bar.progress(75)
                st.session_state.vector_store = vector_store
                st.session_state.image_store = image_store
                st.session_state.documents = documents
                st.session_state.pdf_processed = True
                st.session_state.pdf_name = uploaded_file.name
                
                # Complete
                progress_bar.progress(100)
                status_text.text("✅ Complete!")
                
                # Show success message
                text_count = sum(1 for d in documents if d['type'] == 'text')
                image_count = sum(1 for d in documents if d['type'] == 'image')
                
                st.markdown(f"""
                    <div class="success-box">
                        <h4>✅ PDF Processed Successfully!</h4>
                        <p>
                        📝 Text chunks: <strong>{text_count}</strong><br>
                        🖼️ Images: <strong>{image_count}</strong><br>
                        📊 Total documents: <strong>{len(documents)}</strong>
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Auto-scroll to question section
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")
                st.exception(e)
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                progress_bar.empty()
                status_text.empty()
    else:
        st.markdown("""
            <div class="info-box">
                <p>👆 <strong>Please upload a PDF file to begin</strong></p>
                <p>Supported format: PDF files with text and/or images</p>
            </div>
        """, unsafe_allow_html=True)

# RIGHT COLUMN - Ask Questions
with col2:
    st.header("💬 Step 2: Ask Questions")
    
    if st.session_state.pdf_processed:
        # Question input
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What does the chart show?",
            key="question_input",
            help="Ask anything about the PDF content"
        )
        
        # Predefined questions
        st.subheader("📝 Example Questions")
        example_questions = [
            "What does the chart show about revenue trends?",
            "Summarize the main findings from the document",
            "What visual elements are present in the document?",
            "Which quarter had the highest revenue?",
            "Explain the key data points shown in the images"
        ]
        
        # Create a variable to track selected example
        selected_example = None
        
        cols = st.columns(2)
        for idx, eq in enumerate(example_questions):
            with cols[idx % 2]:
                if st.button(f"📌 {eq[:30]}...", key=f"example_{idx}", use_container_width=True):
                    selected_example = eq
        
        # Use selected example if clicked
        if selected_example:
            question = selected_example
        
        st.divider()
        
        # Ask button
        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            ask_button = st.button("🚀 Get Answer", type="primary", disabled=not question, use_container_width=True)
        with col_btn2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()
        
        if ask_button and question:
            with st.spinner("🤔 Thinking..."):
                try:
                    # Embed query
                    query_embedding = embed_text(question)
                    
                    # Retrieve documents
                    results = st.session_state.vector_store.search(query_embedding, k=k_results)
                    
                    # Show retrieval info immediately
                    st.info(f"Retrieved {len(results)} relevant documents")
                    
                    # Separate text and images
                    text_context = []
                    image_answers = []
                    retrieved_info = []
                    
                    for result in results:
                        doc = result['document']
                        score = result['score']
                        
                        retrieved_info.append(f"[{doc['type'].upper()}] Page {doc['page']} (Score: {score:.3f})")
                        
                        if doc['type'] == 'text':
                            text_context.append(doc['content'])
                        elif doc['type'] == 'image':
                            image = st.session_state.image_store[doc['image_id']]
                            img_answer = answer_with_image(image, question)
                            image_answers.append(f"From image on page {doc['page']}: {img_answer}")
                    
                    # Combine contexts
                    all_context = "\n\n".join(text_context)
                    if image_answers:
                        all_context += "\n\nImage analysis:\n" + "\n".join(image_answers)
                    
                    # Show context being used
                    with st.expander("📝 Context being analyzed"):
                        st.text_area("Context", all_context, height=150)
                    
                    # Generate answer
                    if all_context.strip():
                        answer = generate_text_answer(all_context[:1000], question)  # Limit context length
                        
                        # If answer is empty or too short, create a fallback
                        if not answer or len(answer.strip()) < 10:
                            answer = f"Based on the retrieved documents:\n\n"
                            if text_context:
                                answer += f"Text summary: {text_context[0][:200]}...\n\n"
                            if image_answers:
                                answer += "\n".join(image_answers)
                    else:
                        answer = "No relevant context found to answer the question."
                    
                    # Ensure answer is not empty
                    if not answer or answer.strip() == "":
                        answer = "I found relevant information but couldn't generate a complete answer. Please rephrase your question."
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': question,
                        'answer': answer,
                        'retrieved': retrieved_info
                    })
                    
                    st.success("✅ Answer generated!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)
    
    else:
        st.markdown("""
            <div class="info-box">
                <p>⬅️ <strong>Please upload and process a PDF first!</strong></p>
                <p>Once processed, you can ask questions here.</p>
            </div>
        """, unsafe_allow_html=True)

# Display chat history (full width below)
if st.session_state.chat_history:
    st.markdown("---")
    st.header("💭 Conversation History")
    st.caption(f"Total questions asked: {len(st.session_state.chat_history)}")
    
    # Display in reverse order (newest first)
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        # Question
        st.markdown(f"**🙋 Question #{len(st.session_state.chat_history) - i}:**")
        st.info(chat['question'])
        
        # Answer
        st.markdown("**🤖 Answer:**")
        st.markdown(f"""
            <div class="answer-box">
                {chat['answer']}
            </div>
        """, unsafe_allow_html=True)
        
        # Show retrieval details if enabled
        if show_details:
            with st.expander("📊 View retrieval details"):
                st.markdown("**Retrieved documents:**")
                for info in chat['retrieved']:
                    st.text(f"  • {info}")
        
        st.divider()

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p style='margin: 0;'>🤖 <strong>Multimodal RAG System</strong></p>
        <p style='margin: 0; font-size: 0.9rem;'>Powered by CLIP, BLIP, and FLAN-T5 | 100% Local | No API Keys Required</p>
        <p style='margin: 0; font-size: 0.8rem; margin-top: 0.5rem;'>Built with Streamlit ❤️</p>
    </div>
""", unsafe_allow_html=True)