<template>
  <div class="layout">
    <!-- 1. 顶部指标+搜索 -->
    <header class="top-bar">
      <div class="metrics">
        <span>   ReTopicVis:重构主题关联可视分析系统</span>
        <!-- <span class="divider"/>
        <span>推理后 主题一致性：0.81 | 主题多样性：0.73 | 噪声比：3%</span> -->
      </div>
      <div class="search-box">
        <input
          v-model="searchText"
          placeholder="输入感兴趣文本，回车推荐主题"
          @keyup.enter="onRecommend"
        />
      </div>
    </header>

    <!-- 2. 主面板 -->
    <div class="main">
      <CompactFlowDashboard ref="dashRef" />
      <!-- <FusionView ref="dashRef" /> -->
       <!-- <OrgAnalysisView ref="dashRef" /> -->
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'
import CompactFlowDashboard from './components/CompactFlowDashboard.vue'
// import FusionView from './components/FusionView.vue'
// import OrgAnalysisView from './components/OrgAnalysisView.vue'

const searchText = ref('')
const dashRef = ref(null)

async function onRecommend() {
  if (!searchText.value.trim()) return
  try {
    const { data } = await axios.get('http://localhost:5000/api/recommend_topic', {
      params: { q: searchText.value }
    })
    // 把推荐结果当成“用户手动点击”一样传下去
    // CompactFlowDashboard 里已有 v-model:selected="selectedTopic"
    if (dashRef.value) {
      dashRef.value.selectTopic(data.topic_id)
    }
  } catch (e) {
    alert('推荐失败：' + (e.response?.data?.error || e.message))
  }
}
</script>

<style scoped>
.top-bar {
  height: 40px;
  background: #fff;
  border-bottom: 2px solid #409eff;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 5 15px;
  font-size: 16px;
  color: #2c3e50;
}
.metrics span {
  margin-right: 12px;
  font-size: 20px;
  font-weight: 600;
  color: #2c3e50;
}
.divider {
  display: inline-block;
  width: 1px;
  height: 20px;
  background: #dcdfe6;
  margin: 0 12px;
}
.search-box input {
  width: 260px;
  height: 30px;
  line-height: 32px;
  padding: 0 8px;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
}
</style>