import threading
from pynput import keyboard
from pathlib import Path
import subprocess
import os
import sys
import time
import shutil
from datetime import datetime
import json
from langchain.docstore.document import Document
from src.utils.conversation import saveConversation
import speech_recognition as sr

from src.libs.messages import (print_error_message, print_plugin_message)


_stop_flag = False

def on_press(key):
    global _stop_flag
    try:
        if key.char.lower() in ['q']:
            _stop_flag = True
            return False
    except AttributeError:
        if key == keyboard.Key.space:
            _stop_flag = True
            return False


def _key_listener():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


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
    """
    Plays an audio file using ffplay and allows the user to stop playback with SPACEBAR or Q.
    """
    global _stop_flag
    _stop_flag = False

    if not filepath.exists():
        print_error_message(f"Audio file not found: {filepath}")
        return {"success": False, "message": "Audio file not found."}

    print_plugin_message("PLAYING... [Press SPACEBAR or Q to stop playback]")

    proc = subprocess.Popen(
        ['ffplay', '-nodisp', '-autoexit', str(filepath)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    listener = threading.Thread(target=_key_listener, daemon=True)
    listener.start()

    while proc.poll() is None:
        if _stop_flag:
            print_plugin_message("STOPPING...")
            proc.terminate()
            break
        time.sleep(0.1)

    proc.wait()
    return {"success": True, "message": "Audio playback successful."}


def _process_and_play_text(text_to_speak, current_memory_path, piper_executable, model_path):
    """Synthesizes text to audio and plays the file."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    audio_dir = current_memory_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    main_output_filepath = audio_dir / "aeon_output.wav"
    timestamped_copy_filepath = audio_dir / f"aeon_{timestamp}.wav"

    print_plugin_message("Synthesizing audio with Piper...")
    piper_command = [
        piper_executable,
        '--model', str(model_path),
        '--output_file', str(main_output_filepath)
    ]
    
    try:
        subprocess.run(
            piper_command,
            input=text_to_speak.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True
        )
    except FileNotFoundError:
        print_error_message(f"The '{piper_executable}' executable was not found. Please ensure Piper is installed and in your system's PATH.")
        return {"success": False, "message": f"Piper executable not found."}
    except subprocess.CalledProcessError as e:
        print_error_message(f"Piper failed with error: {e.stderr.decode()}")
        return {"success": False, "message": f"Piper failed: {e.stderr.decode()}"}
    except Exception as e:
        print_error_message(f"An error occurred during TTS operation: {e}")
        return {"success": False, "message": f"TTS failed: {e}"}

    shutil.copy(main_output_filepath, timestamped_copy_filepath)
    
    print_plugin_message(f"[AEON]: {text_to_speak}")
    print_plugin_message(f"AUDIO SAVED: {timestamped_copy_filepath.resolve()}")
    
    return _play_audio_file(main_output_filepath)


def _listen_and_transcribe():
    """Listens for user speech and transcribes it to text."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print_plugin_message("Listening...")
        r.adjust_for_ambient_noise(source)
        try:
            audio = r.listen(source, phrase_time_limit=5)
            print_plugin_message("Transcribing...")
            text = r.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            print_error_message("Could not understand audio.")
            return None
        except sr.RequestError as e:
            print_error_message(f"Could not request results from Google Speech Recognition service; {e}")
            return None

def run_plugin(args: str, **kwargs) -> dict:
    plugin_config = kwargs.get('plugin_config')
    plugin_name = plugin_config.get("plugin_name")
    vectorstore = kwargs.get('vectorstore')
    text_splitter = kwargs.get('text_splitter')
    llama_embeddings = kwargs.get('llama_embeddings')
    conversation_filename = kwargs.get('conversation_filename')
    current_memory_path = kwargs.get('current_memory_path')
    
    rag_chain = kwargs.get("rag_chain")

    args_list = args.split(" ", 1)
    command = args_list[0].lower()
    prompt = args_list[1] if len(args_list) > 1 else ""

    if command == "/talk":
        print_plugin_message("Entering conversational mode. Say 'stop' or 'goodbye' to exit.")
        while True:
            user_input = _listen_and_transcribe()
            if not user_input:
                continue

            print_plugin_message(f"You said: {user_input}")
            if user_input.lower() in ["stop", "goodbye", "exit"]:
                print_plugin_message("Exiting conversational mode.")
                return {"success": True, "message": "Exited conversational mode."}

            if not rag_chain:
                print_error_message("RAG system not initialized.")
                return {"success": False, "message": "RAG system not initialized."}

            try:
                print_plugin_message("Generating response using RAG chain...")
                result = rag_chain.invoke(user_input)
                aeon_response_text = result.get("answer", "No answer found.")

                if not aeon_response_text:
                    print_plugin_message("RAG system returned an empty response. Cannot generate audio.")
                    continue

                _ingest_conversation_turn(
                    user_input,
                    aeon_response_text,
                    vectorstore,
                    text_splitter,
                    llama_embeddings
                )

                saveConversation(
                    user_input,
                    aeon_response_text,
                    plugin_name,
                    current_memory_path,
                    conversation_filename
                )
                
                plugin_dir = Path(__file__).parent
                model_path = plugin_dir / "model" / "en_US-kathleen-low.onnx"
                piper_executable = "piper"

                _process_and_play_text(aeon_response_text, current_memory_path, piper_executable, model_path)

            except Exception as e:
                print_error_message(f"An error occurred: {e}")
                continue

    elif command == "/replay":
        main_output_filepath = current_memory_path / "audio" / "aeon_output.wav"
        if not main_output_filepath.exists():
            print_error_message("No audio file found to replay. Please generate a response first.")
            return {"success": False, "message": "No audio file to replay."}
        
        return _play_audio_file(main_output_filepath)
        
    else: # Default behavior for regular text-to-speech
        if not args:
            print_error_message(f"Usage: /say <PROMPT>")
            print_plugin_message(f"Or use /say /replay to listen to the last generated audio.")
            return {"success": False, "message": "No prompt provided."}

        if not rag_chain:
            print_error_message("RAG system not initialized.")
            return {"success": False, "message": "RAG system not initialized."}

        try:
            print_plugin_message("Generating response using RAG chain...")
            result = rag_chain.invoke(args)
            aeon_response_text = result.get("answer", "No answer found.")

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
            
            plugin_dir = Path(__file__).parent
            model_path = plugin_dir / "model" / "en_US-kathleen-low.onnx"
            piper_executable = "piper"

            return _process_and_play_text(aeon_response_text, current_memory_path, piper_executable, model_path)

        except Exception as e:
            print_error_message(f"An error occurred: {e}")
            return {"success": False, "message": f"An unexpected error occurred: {e}"}
