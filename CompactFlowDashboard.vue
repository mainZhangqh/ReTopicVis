<template>
  <div class="dashboard-wrap">
    <!-- 第一列：FusionView（跨第1-3行） -->
    <div class="panel fusion-panel">
      <FusionView v-model:selected="selectedTopic" />
    </div>

    <!-- 第二列：合著者视图（跨第1-2行） -->
    <div class="panel coauthor-panel">
      <Collaboration 
        :filter-topic="normalizedSelectedTopic" 
        :filter-org="selectedOrg"
        @clear-org-filter="handleClearOrgFilter"
      />
    </div>

    <!-- 第二列：层次演化树图（第3行） -->
    <div class="panel tree-panel">
      <VerticalTreeChart
        title="推理后：层次演化树图"
        apiUrl="http://localhost:5000/api/topic_tree_new"
        v-model:selected="selectedTopic"
        :similarityData="similarityData"
      />
    </div>

    <!-- 第三列：词云（第1行） -->
    <div class="panel wordcloud-panel">
      <WordCloudView
        :selectedTopic="normalizedSelectedTopic"
        @word-click="handleWordClick"
        @topic-click="handleTopicClickFromWord"
      />
    </div>

    <!-- 第四行（整行）：机构主题分析视图 -->
    <div class="panel org-analysis-panel">
      <OrgAnalysisView 
        :selectedTopic="normalizedSelectedTopic"
        @org-click="handleOrgClick"
      />
    </div>

    <!-- 第五行（整行）：底部表格区域 -->
    <div class="panel detail-panel">
      <PaperTable 
        :topic="normalizedSelectedTopic" 
        :author="selectedAuthor"
        :org="selectedOrg"
        @clear-org-filter="handleClearOrgFilter"
      />
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch, computed } from 'vue'
import axios from 'axios'
import VerticalTreeChart from './VerticalTreeChart.vue'
import Collaboration from './Collaboration.vue'
import PaperTable from './PaperTable.vue'
import FusionView from './FusionView.vue'
import WordCloudView from './WordCloudView.vue'
import OrgAnalysisView from './OrgAnalysisView.vue'

/* ---------- 状态管理 ---------- */
const selectedTopic = ref(null)          // 当前选中的主题 ID（可能为数组）
const similarityData = ref({})           // 存储主题相似度矩阵
const selectedAuthor = ref(null)         // 当前选中的作者
const selectedOrg = ref(null)            // 当前选中的机构

/* ---------- 计算属性：规范化主题ID ---------- */
// 确保传递给子组件的主题ID始终是单个数字或null
const normalizedSelectedTopic = computed(() => {
  const topic = selectedTopic.value
  if (Array.isArray(topic)) {
    // 如果是数组，取第一个元素
    return topic[0] || null
  }
  return topic
})

/* ---------- 数据加载 ---------- */
const loadSimilarityData = async () => {
  try {
    const res = await axios.get('http://localhost:5000/api/collaboration')
    const data = res.data
    const combinedSimilarity = {}
    
    if (data && data.nodes) {
      data.nodes.forEach(node => {
        const topicId = node.id
        combinedSimilarity[topicId] = {}
        data.links.forEach(link => {
          if (link.source === topicId) combinedSimilarity[topicId][link.target] = link.weight
          if (link.target === topicId) combinedSimilarity[topicId][link.source] = link.weight
        })
      })
    }
    similarityData.value = combinedSimilarity
  } catch (error) {
    console.error('加载相似度数据失败:', error)
    similarityData.value = {}
  }
}

/* ===== 供父组件主动调用 ===== */
function selectTopic(id) {
  // 规范化输入的主题ID
  if (Array.isArray(id)) {
    selectedTopic.value = id[0] || null
  } else {
    selectedTopic.value = id
  }
  console.log('Dashboard: 选择主题', selectedTopic.value)
}
defineExpose({ selectTopic })   // 让 ref 能调到

/* ---------- 机构点击处理 ---------- */
const handleOrgClick = (orgName) => {
  console.log('选中机构:', orgName)
  selectedOrg.value = orgName
  
  // 触发全局事件，通知其他组件
  window.dispatchEvent(new CustomEvent('org-filter-change', { 
    detail: orgName 
  }))
  
  // 也可以直接在这里更新子组件（备选方案）
  // 但更好的方式是通过 props 传递
}

/* ---------- 清除机构筛选 ---------- */
const handleClearOrgFilter = () => {
  console.log('清除机构筛选')
  selectedOrg.value = null
  
  // 触发全局事件，通知其他组件
  window.dispatchEvent(new CustomEvent('org-filter-change', { 
    detail: null 
  }))
}

/* ---------- 词云事件处理 ---------- */
const handleWordClick = (wordData) => {
  console.log('点击了词云关键词:', wordData)
}

const handleTopicClickFromWord = (topicId) => {
  console.log('从词云点击了主题:', topicId)
  // 规范化处理：确保存储的是单个值
  if (Array.isArray(topicId)) {
    selectedTopic.value = topicId[0] || null
  } else {
    selectedTopic.value = topicId
  }
}

/* ---------- 生命周期 ---------- */
onMounted(() => {
  loadSimilarityData()
  
  // 监听机构筛选变化事件
  window.addEventListener('org-filter-change', (e) => {
    console.log('收到机构筛选变化事件:', e.detail)
    selectedOrg.value = e.detail
  })
})

/* ---------- 事件监听 ---------- */
// 监听作者选择
window.addEventListener('author-change', (e) => {
  selectedAuthor.value = e.detail || null
  console.log('Dashboard: 选中作者', selectedAuthor.value)
})

// 监听主题变化
watch(selectedTopic, (newVal) => {
  if (newVal !== null) {
    console.log(`Dashboard: 主题更新为`, newVal)
    console.log(`规范化后的主题: ${normalizedSelectedTopic.value}`)
  }
})

// 监听机构变化
watch(selectedOrg, (newVal) => {
  if (newVal) {
    console.log(`Dashboard: 机构筛选更新为 ${newVal}`)
  } else {
    console.log('Dashboard: 清空机构筛选')
  }
})

// 监听规范化后的主题变化（用于调试）
watch(normalizedSelectedTopic, (newVal) => {
  if (newVal !== null) {
    console.log(`Dashboard: 规范化主题 T${newVal} 传递给子组件`)
  }
})
</script>

<style scoped>
.dashboard-wrap {
  display: grid;
  /* 三列宽度比例 */
  grid-template-columns: 1fr 1fr 0.55fr;
  /* 五行：重新分配行高 */
  grid-template-rows: 1.2fr 0.5fr 0.6fr 1fr 1.2fr;
  gap: 6px;
  height: calc(100vh + 30px);
  width: 100%;
  padding: 5px;
  background: #f0f2f5;
  box-sizing: border-box;
  overflow: hidden;
}

/* 通用面板样式 */
.panel {
  background: #fff;
  border-radius: 8px;
  border: 1px solid #dcdfe6;
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
  position: relative;
}

/* 第一列：FusionView面板（跨第1-3行） */
.fusion-panel {
  grid-column: 1;
  grid-row: 1 / span 4; /* 跨第1-3行 */
}

/* 第二列：合著者面板（跨第1-2行） */
.coauthor-panel {
  grid-column: 2;
  grid-row: 1 / span 3; /* 跨第1-2行 */
}

/* 第二列：层次树面板（第3行） */
.tree-panel {
  grid-column: 3;
  grid-row: 2 / span 2; /* 第3行 */
}

/* 第三列：词云面板（第1行） */
.wordcloud-panel {
  grid-column: 3;
  grid-row: 1; /* 第1行 */
}

/* 第四行：机构主题分析视图（跨所有三列） */
.org-analysis-panel {
  grid-column: 2 / span 2; /* 跨第1-3列 */
  grid-row: 4; /* 第4行 */
}

/* 第五行：底部表格区域（跨所有三列） */
.detail-panel {
  grid-column: 1 / span 3; /* 占所有三列 */
  grid-row: 5; /* 第5行 */
}

/* 面板标题样式 */
.panel-title {
  padding: 12px 16px;
  margin: 0;
  font-size: 15px;
  font-weight: 600;
  color: #303133;
  border-bottom: 1px solid #e8e8e8;
  background: #fafafa;
}

/* 面板内容区域 */
.panel-content {
  flex: 1;
  overflow: auto;
  padding: 16px;
}

/* 响应式调整 */
@media (max-width: 1600px) {
  .dashboard-wrap {
    grid-template-columns: 1.1fr 1.2fr 0.9fr;
  }
}

@media (max-width: 1400px) {
  .dashboard-wrap {
    grid-template-columns: 1.1fr 1.3fr 0.8fr;
    grid-template-rows: 0.9fr 0.9fr 0.9fr 1.8fr 1.5fr;
  }
}

@media (max-width: 1200px) {
  .dashboard-wrap {
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr 1fr 2fr 1.5fr;
  }
  
  /* 调整第三列在平板上的位置 */
  .wordcloud-panel {
    grid-column: 1;
    grid-row: 4;
  }
  
  .org-analysis-panel {
    grid-column: 1 / span 2;
    grid-row: 5;
  }
  
  .detail-panel {
    grid-column: 1 / span 2;
    grid-row: 6;
  }
}

@media (max-width: 768px) {
  .dashboard-wrap {
    grid-template-columns: 1fr;
    grid-template-rows: repeat(7, auto);
    height: auto;
    min-height: 100vh;
  }
  
  /* 将所有面板堆叠显示 */
  .fusion-panel,
  .coauthor-panel,
  .tree-panel,
  .wordcloud-panel,
  .org-analysis-panel,
  .detail-panel {
    grid-column: 1;
  }
  
  .fusion-panel {
    grid-row: 1;
    height: 400px;
  }
  
  .coauthor-panel {
    grid-row: 2;
    height: 350px;
  }
  
  .tree-panel {
    grid-row: 3;
    height: 300px;
  }
  
  .wordcloud-panel {
    grid-row: 4;
    height: 300px;
  }
  
  .org-analysis-panel {
    grid-row: 5;
    min-height: 400px;
  }
  
  .detail-panel {
    grid-row: 6;
    min-height: 350px;
  }
}

/* 面板悬停效果 */
.panel:hover {
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  transition: box-shadow 0.3s;
}

/* 加载状态 */
.panel-loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #909399;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid #f3f3f3;
  border-top: 3px solid #409eff;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 16px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* 空状态 */
.panel-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #909399;
  padding: 20px;
  text-align: center;
}

.empty-icon {
  font-size: 48px;
  margin-bottom: 16px;
  opacity: 0.5;
}

.empty-text {
  font-size: 14px;
  margin-bottom: 8px;
}

.empty-subtext {
  font-size: 12px;
  color: #999;
}
</style>