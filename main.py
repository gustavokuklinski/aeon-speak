import sys
from pathlib import Path
from gtts import gTTS
import pygame
import readchar
import time
import shutil
from datetime import datetime
from langchain.docstore.document import Document
from src.utils.conversation import saveConversation

from src.libs.messages import (print_error_message, print_plugin_message)

def _ingest_conversation_turn(user_input, aeon_output, vectorstore, text_splitter, llama_embeddings):
    try:
        conversation_text = f"{user_input}\n\n{aeon_output}"
        
        conversation_document = Document(
            page_content=conversation_text,
            metadata={"source": "speak"}
        )
        
        docs = text_splitter.split_documents([conversation_document])
        success, failed = 0, 0
        for i, chunk in enumerate(docs, start=1):
            try:
                vectorstore.add_documents([chunk])
                success += 1
            except Exception as e:
                failed += 1
                print_error_message(f" Failed on chunk {i}: {e}")
    except Exception as e:
        print_error_message(f"Failed to ingest conversation turn: {e}")

def _play_audio_file(filepath: Path):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(str(filepath))
        pygame.mixer.music.play()
        
        print_plugin_message("PLAYING... [Press SPACE to stop playback and back to prompt]")
        
        while pygame.mixer.music.get_busy():
            try:
                key = readchar.readkey()
                if key == ' ':
                    print_plugin_message("STOPPING...")
                    pygame.mixer.music.stop()
            except UnicodeDecodeError:
                pass
            time.sleep(0.1)

        print_plugin_message(f"LISTEN AGAIN AT: {filepath}.")
        return {"success": True, "message": "Audio playback successful."}

    except Exception as e:
        print_error_message(f"Could not play audio using Pygame. Error: {e}")
        return {"success": False, "message": f"Audio playback failed: {e}"}


def run_plugin(args: str, **kwargs) -> dict:
    plugin_config = kwargs.get('plugin_config')
    plugin_name = plugin_config.get("plugin_name")
    vectorstore = kwargs.get('vectorstore')
    text_splitter = kwargs.get('text_splitter')
    llama_embeddings = kwargs.get('llama_embeddings')
    conversation_filename = kwargs.get('conversation_filename')
    current_memory_path = kwargs.get('current_memory_path')
    current_chat_history=kwargs.get("current_chat_history")
    
    args_list = args.split(" ", 1)
    command = args_list[0].lower()
    prompt = args_list[1] if len(args_list) > 1 else ""

    if not args:
        print_error_message(f"Usage: /{plugin_name} <PROMPT>")
        print_plugin_message(f"Or use /{plugin_name} replay to listen to the last generated audio.")
        return {"success": False, "message": "No prompt provided."}
    
    if command == "/replay":
        main_output_filepath = current_memory_path / "audio" / "aeon_output.mp3"
        if main_output_filepath.exists():
            return _play_audio_file(main_output_filepath)
        else:
            print_error_message("No audio file found to replay. Please generate a response first.")
            return {"success": False, "message": "No audio file to replay."}

    rag_chain = kwargs.get("rag_chain")
    if not rag_chain:
        print_error_message("RAG system not initialized.")
        return {"success": False, "message": "RAG system not initialized."}

    try:
        print_plugin_message("Generating response using RAG chain...")
        result = rag_chain.invoke(args)
        aeon_response_text = result.get("answer", "No answer found.")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        audio_dir = current_memory_path / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        main_output_filepath = audio_dir / "aeon_output.mp3"
        timestamped_copy_filepath = audio_dir / f"aeon_{timestamp}.mp3"


        if not aeon_response_text:
            print_plugin_message("RAG system returned an empty response. Cannot generate audio.")
            return {"success": False, "message": "RAG system returned an empty response."}

        _ingest_conversation_turn(
            args,
            aeon_response_text,
            vectorstore,
            text_splitter,
            llama_embeddings
        )

        saveConversation(
            args,
            aeon_response_text,
            plugin_name,
            current_memory_path,
            conversation_filename
        )

        current_chat_history.append(
            {"user": args, plugin_name: aeon_response_text, "source": f"{plugin_name} /aeon_{timestamp}.mp3"}
        )

        tts = gTTS(text=aeon_response_text, lang='en')
        tts.save(main_output_filepath)
        
        shutil.copy(main_output_filepath, timestamped_copy_filepath)
        
        print_plugin_message(f"[AEON]: {aeon_response_text}")
        print_plugin_message(f"AUDIO SAVED: {timestamped_copy_filepath.resolve()}")

        return _play_audio_file(main_output_filepath)

    except Exception as e:
        print_error_message(f"An error occurred during gTTS operation: {e}")
        return {"success": False, "message": f"gTTS failed: {e}"}

