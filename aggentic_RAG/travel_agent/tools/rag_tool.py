"""
RAG检索工具
"""
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Optional
import os
from pathlib import Path
import hashlib
import uuid

from ..config.settings import (
    DASHSCOPE_API_KEY,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP,
    RAG_SEARCH_K,
    RAG_BATCH_SIZE,
)


class TravelRAG:
    """旅游攻略RAG系统"""
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_api_key: Optional[str] = None
    ):
        self.persist_directory = persist_directory or str(CHROMA_PERSIST_DIR)
        self.embedding_api_key = embedding_api_key or DASHSCOPE_API_KEY
        
        # 用于记录已导入的文档ID，避免重复
        self.imported_ids = set()
        
        # 初始化Embeddings
        self.embeddings = DashScopeEmbeddings(
            model=EMBEDDING_MODEL,
            dashscope_api_key=self.embedding_api_key
        )
        
        # 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAG_CHUNK_SIZE,
            chunk_overlap=RAG_CHUNK_OVERLAP
        )
        
        # 初始化向量数据库
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """初始化向量数据库"""
        if os.path.exists(self.persist_directory):
            print(f"✅ 加载现有向量数据库: {self.persist_directory}")
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="travel_knowledge"
            )
        else:
            print(f"⚠️ 向量数据库不存在，请先加载文档: {self.persist_directory}")
            print("   使用 build_knowledge_base() 方法创建知识库")
            self.vector_store = None
        
        # 创建检索器
        if self.vector_store:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": RAG_SEARCH_K}
            )
            # 加载已存在的ID
            try:
                existing_data = self.vector_store.get()
                if existing_data and 'ids' in existing_data:
                    self.imported_ids = set(existing_data['ids'])
                    print(f"  📊 已加载 {len(self.imported_ids)} 条现有数据")
            except Exception as e:
                print(f"  ⚠️ 无法加载现有ID: {e}")
    
    def load_documents(self, source_path: str, file_type: str = "auto") -> List[Document]:
        """加载文档 - 支持多种格式
        
        支持的格式：
        - TXT: 纯文本文件
        - MD: Markdown文件
        - PDF: PDF文档（需要pypdf）
        - CSV: CSV数据文件
        - Directory: 自动扫描目录下所有支持的格式
        """
        documents = []
        
        try:
            path = Path(source_path)
            
            # 自动检测文件类型
            if file_type == "auto":
                if path.is_dir():
                    file_type = "directory"
                else:
                    ext = path.suffix.lower()
                    type_map = {
                        '.txt': 'txt',
                        '.md': 'md',
                        '.pdf': 'pdf',
                        '.csv': 'csv'
                    }
                    file_type = type_map.get(ext, 'txt')
            
            # 加载单个TXT文件
            if file_type == "txt":
                loader = TextLoader(source_path, encoding="utf-8")
                documents = loader.load()
                
            # 加载单个Markdown文件
            elif file_type == "md":
                loader = TextLoader(source_path, encoding="utf-8")
                documents = loader.load()
                
            # 加载单个PDF文件
            elif file_type == "pdf":
                try:
                    loader = PyPDFLoader(source_path)
                    documents = loader.load()
                except ImportError:
                    print("⚠️ 需要安装pypdf: pip install pypdf")
                    raise
                    
            # 加载单个CSV文件
            elif file_type == "csv":
                try:
                    loader = CSVLoader(source_path, encoding="utf-8")
                    documents = loader.load()
                except ImportError:
                    print("⚠️ 需要安装csv支持")
                    raise
                    
            # 加载整个目录
            elif file_type == "directory":
                loaders = []
                
                # TXT文件
                try:
                    txt_loader = DirectoryLoader(
                        source_path,
                        glob="**/*.txt",
                        loader_cls=TextLoader,
                        loader_kwargs={"encoding": "utf-8"},
                        show_progress=True
                    )
                    loaders.append(txt_loader)
                except Exception as e:
                    print(f"⚠️ TXT加载器初始化失败: {e}")
                
                # Markdown文件
                try:
                    md_loader = DirectoryLoader(
                        source_path,
                        glob="**/*.md",
                        loader_cls=TextLoader,
                        loader_kwargs={"encoding": "utf-8"},
                        show_progress=True
                    )
                    loaders.append(md_loader)
                except Exception as e:
                    print(f"⚠️ MD加载器初始化失败: {e}")
                
                # PDF文件
                try:
                    pdf_loader = DirectoryLoader(
                        source_path,
                        glob="**/*.pdf",
                        loader_cls=PyPDFLoader,
                        show_progress=True
                    )
                    loaders.append(pdf_loader)
                except ImportError:
                    print("⚠️ 跳过PDF文件（需要安装pypdf: pip install pypdf）")
                except Exception as e:
                    print(f"⚠️ PDF加载器初始化失败: {e}")
                
                # CSV文件
                try:
                    csv_loader = DirectoryLoader(
                        source_path,
                        glob="**/*.csv",
                        loader_cls=CSVLoader,
                        show_progress=True
                    )
                    loaders.append(csv_loader)
                except ImportError:
                    print("⚠️ 跳过CSV文件")
                except Exception as e:
                    print(f"⚠️ CSV加载器初始化失败: {e}")
                
                # 执行所有加载器
                for loader in loaders:
                    try:
                        docs = loader.load()
                        documents.extend(docs)
                        print(f"  ✅ 成功加载 {len(docs)} 个文档片段")
                    except Exception as e:
                        print(f"  ⚠️ 部分文件加载失败: {e}")
                        continue
                        
            else:
                raise ValueError(f"不支持的文件类型: {file_type}")
            
            print(f"\n✅ 总共成功加载 {len(documents)} 个文档")
            return documents
            
        except Exception as e:
            print(f"❌ 文档加载失败: {e}")
            raise
    
    @staticmethod
    def generate_doc_id(doc: Document, chunk_index: int = 0) -> str:
        """基于文档内容和来源生成稳定UUID
        
        Args:
            doc: 文档对象
            chunk_index: 分块索引，用于同一文档的不同分块
            
        Returns:
            稳定的UUID字符串
        """
        content = doc.page_content
        source = doc.metadata.get('source', '')
        # 用来源+块索引+内容前100字符生成hash
        hash_input = f"{source}:{chunk_index}:{content[:100]}"
        hash_hex = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
        # 将MD5 hash转换为UUID
        return str(uuid.UUID(hash_hex))
    
    def build_knowledge_base(
        self,
        source_path: str,
        file_type: str = "directory",
        force_recreate: bool = False
    ):
        """构建知识库"""
        print(f"\n📂 正在加载文档: {source_path}")
        
        # 加载文档
        documents = self.load_documents(source_path, file_type)
        
        # 分割文档
        split_docs = self.text_splitter.split_documents(documents)
        print(f"✅ 文档分割完成，共 {len(split_docs)} 个块")
        
        # 生成UUID
        print(f"🆔 正在生成UUID...")
        doc_ids = [self.generate_doc_id(doc, idx) for idx, doc in enumerate(split_docs)]
        
        # 检查重复
        if not force_recreate and self.imported_ids:
            new_ids = [doc_id for doc_id in doc_ids if doc_id not in self.imported_ids]
            duplicate_count = len(doc_ids) - len(new_ids)
            if duplicate_count > 0:
                print(f"  ⚠️ 发现 {duplicate_count} 个重复文档块，将跳过")
                # 过滤出新文档
                filtered_docs = [doc for doc, doc_id in zip(split_docs, doc_ids) if doc_id not in self.imported_ids]
                filtered_ids = new_ids
                split_docs = filtered_docs
                doc_ids = filtered_ids
                print(f"  ✅ 实际导入 {len(split_docs)} 个新文档块")
        
        if len(split_docs) == 0:
            print(f"⚠️ 没有新文档需要导入")
            return
        
        # 创建或更新向量数据库
        if force_recreate and os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory)
            self.imported_ids.clear()
        
        print(f"📊 正在创建向量数据库...")
        
        # 批量处理，使用配置的batch_size
        batch_size = RAG_BATCH_SIZE
        for i in range(0, len(split_docs), batch_size):
            batch = split_docs[i:i+batch_size]
            batch_ids = doc_ids[i:i+batch_size]
            
            if i == 0 and (force_recreate or not self.vector_store):
                self.vector_store = Chroma.from_documents(
                    documents=batch,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                    collection_name="travel_knowledge",
                    ids=batch_ids  # 指定UUID
                )
            else:
                self.vector_store.add_documents(documents=batch, ids=batch_ids)
            
            # 更新已导入ID集合
            self.imported_ids.update(batch_ids)
            
            print(f"  处理进度: {min(i+batch_size, len(split_docs))}/{len(split_docs)} 批")
        
        print(f"✅ 知识库构建完成！")
        print(f"📊 总数据量: {len(self.imported_ids)} 条")
        
        # 重新初始化检索器
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RAG_SEARCH_K}
        )
    
    def delete_by_source(self, source_path: str) -> int:
        """根据来源文件删除数据
        
        Args:
            source_path: 文件路径
            
        Returns:
            删除的数据条数
        """
        if not self.vector_store:
            print("⚠️ 向量数据库未初始化")
            return 0
        
        try:
            # 查找要删除的文档
            results = self.vector_store.get(
                where={"source": source_path}
            )
            
            if results and results['ids']:
                delete_count = len(results['ids'])
                self.vector_store.delete(ids=results['ids'])
                # 从集合中移除
                self.imported_ids -= set(results['ids'])
                print(f"✅ 已删除 {delete_count} 条数据（来源: {source_path}）")
                return delete_count
            else:
                print(f"⚠️ 未找到来源为 {source_path} 的数据")
                return 0
        except Exception as e:
            print(f"❌ 删除失败: {e}")
            return 0
    
    def get_stats(self) -> dict:
        """获取知识库统计信息"""
        if not self.vector_store:
            return {"total": 0, "sources": []}
        
        try:
            results = self.vector_store.get()
            sources = set([metadata.get('source', '未知') for metadata in results.get('metadatas', [])])
            return {
                "total": len(results.get('ids', [])),
                "sources": list(sources)
            }
        except Exception as e:
            print(f"⚠️ 获取统计信息失败: {e}")
            return {"total": 0, "sources": []}
    
    async def search(self, query: str, k: Optional[int] = None) -> str:
        """检索旅游攻略"""
        import asyncio
        
        if not self.vector_store:
            return "知识库未初始化，请先构建知识库"
        
        k = k or RAG_SEARCH_K
        
        try:
            # 使用 asyncio.to_thread 将同步调用移到线程池
            docs = await asyncio.to_thread(
                self.vector_store.similarity_search,
                query,
                k=k
            )
            
            if not docs:
                return "未找到相关旅游攻略"
            
            results = []
            for i, doc in enumerate(docs, 1):
                content = doc.page_content[:300]
                source = doc.metadata.get("source", "未知")
                results.append(f"[{i}] 来源: {source}\n{content}")
            
            return "\n\n".join(results)
        except Exception as e:
            return f"检索失败: {str(e)}"


# 全局RAG实例
_rag_instance = None


def get_rag_instance() -> TravelRAG:
    """获取全局RAG实例"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = TravelRAG()
    return _rag_instance
