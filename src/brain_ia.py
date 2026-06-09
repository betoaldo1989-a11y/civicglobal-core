import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

class AsesorCiudadanoIA:
    def __init__(self, db_path="./chroma_db", model_name="llama3"):
        """
        Inicializa el Cerebro Digital descentralizado de CivicGlobal.
        Configura la base vectorial local y el modelo LLM P2P.
        """
        self.db_path = db_path
        self.model_name = model_name
        
        print("[*] Conectando con modelos locales de Ollama...")
        # Inicialización de motores locales e independientes
        self.llm = Ollama(model=self.model_name)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vector_store = None

    def indexar_leyes_locales(self, carpeta_leyes="leyes_genesis/"):
        """
        Lee los archivos de leyes en texto plano, los fragmenta matemáticamente 
        y genera los embeddings en la base de datos persistente local.
        """
        if not os.path.exists(carpeta_leyes):
            raise FileNotFoundError(f"La carpeta {carpeta_leyes} no existe.")
            
        print(f"[*] Leyendo legislación local desde: {carpeta_leyes}")
        documentos = []
        
        for archivo in os.listdir(carpeta_leyes):
            if archivo.endswith(".txt"):
                ruta_completa = os.path.join(carpeta_leyes, archivo)
                with open(ruta_completa, "r", encoding="utf-8") as f:
                    documentos.append(f.read())
        
        if not documentos:
            print("[!] No se encontraron archivos de texto para indexar.")
            return

        # Fragmentación matemática estricta para el RAG
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        textos_fragmentados = text_splitter.create_documents(documentos)
        
        print(f"[*] Creando base vectorial con {len(textos_fragmentados)} fragmentos...")
        self.vector_store = Chroma.from_documents(
            documents=textos_fragmentados,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        print("[✓] Base de datos de leyes indexada y protegida localmente.")

    def generar_defensa_automatica(self, denuncia_ciudadana):
        """
        Recibe un caso de abuso, busca los fundamentos legales 
        en la base vectorial y genera una estructura formal de defensa.
        """
        print("[*] Procesando denuncia ciudadana y buscando fundamentos...")
        
        if not self.vector_store:
            # Si no se ha indexado en la sesión actual, intenta cargar la DB existente
            if os.path.exists(self.db_path):
                self.vector_store = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
            else:
                return "Error: No hay base de datos de leyes cargada para fundamentar la defensa."

        # Configuración del motor de búsqueda y generación (RAG)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever
        )
        
        # Prompt estructurado bajo la mística del protocolo
        instrucción_maestra = f"""
        Actúa como un algoritmo de defensa ciudadana incorruptible para el protocolo CivicGlobal.
        Analiza la siguiente denuncia y redacta una defensa formal y estricta en español.
        Es obligatorio que tu respuesta contenga exclusivamente las siguientes secciones en mayúsculas:
        1. PROEMIO (A quién va dirigido y quién defiende).
        2. HECHOS (Narración clara del abuso reportado).
        3. FUNDAMENTOS LEGALES (Sustento basado en el contexto recuperado).
        4. PUNTOS PETITORIOS (Qué se exige concretamente para reparar el daño).
        
        Denuncia a procesar: {denuncia_ciudadana}
        """
        
        respuesta = qa_chain.run(instrucción_maestra)
        return respuesta

if __name__ == "__main__":
    # Inicialización de prueba del nodo local
    print("=== NODO CIVICGLOBAL GENERACIÓN GÉNESIS ===")
    try:
        asesor = AsesorCiudadanoIA()
        print("[✓] Nodo de Inteligencia Artificial CivicGlobal inicializado correctamente.")
    except Exception as e:
        print(f"[!] Error al iniciar el nodo local: {e}")
