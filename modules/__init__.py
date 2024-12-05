from .data_processor import DataProcessor
from .hybrid_retriever import HybridRetriever
from .conversation_manager import ConversationManager

__all__ = ['DataProcessor', 'HybridRetriever', 'ConversationManager']RTF(Real-Time Factor, 실시간 처리 비율) 평가 방법에 대해 설명해드리겠습니다:

1. **RTF 기본 개념**
```python
def calculate_rtf(processing_time, audio_duration):
    """
    RTF = 처리 시간 / 오디오 길이
    RTF < 1: 실시간 처리 가능
    RTF > 1: 실시간 처리 불가
    """
    rtf = processing_time / audio_duration
    return rtf
```

2. **세부 RTF 측정 구현**
```python
def measure_rtf(model, audio_file):
    # 오디오 로드
    audio, sr = sf.read(audio_file)
    audio_duration = len(audio) / sr
    
    # 처리 시간 측정
    start_time = time.time()
    _ = model.process(audio)
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    rtf = calculate_rtf(processing_time, audio_duration)
    return {
        'rtf': rtf,
        'processing_time': processing_time,
        'audio_duration': audio_duration
    }
```

3. **배치 처리 RTF**
```python
def batch_rtf_measurement(model, audio_files):
    results = []
    
    for file in audio_files:
        result = measure_rtf(model, file)
        results.append({
            'file': file,
            'rtf': result['rtf'],
            'is_realtime': result['rtf'] < 1
        })
    
    return results
```

4. **RTF 모니터링 시스템**
```python
class RTFMonitor:
    def __init__(self):
        self.measurements = []
        
    def start_measurement(self):
        self.start_time = time.time()
        
    def end_measurement(self, audio_duration):
        processing_time = time.time() - self.start_time
        rtf = calculate_rtf(processing_time, audio_duration)
        
        self.measurements.append({
            'rtf': rtf,
            'timestamp': datetime.now(),
            'audio_duration': audio_duration
        })
        
    def get_statistics(self):
        rtf_values = [m['rtf'] for m in self.measurements]
        return {
            'mean_rtf': np.mean(rtf_values),
            'std_rtf': np.std(rtf_values),
            'min_rtf': np.min(rtf_values),
            'max_rtf': np.max(rtf_values)
        }
```

5. **RTF 최적화 모니터링**
```python
def rtf_optimization_analysis(model, audio_file, batch_sizes=[1, 2, 4, 8, 16]):
    results = []
    
    for batch_size in batch_sizes:
        rtf = measure_rtf_with_batch(model, audio_file, batch_size)
        results.append({
            'batch_size': batch_size,
            'rtf': rtf,
            'throughput': 1/rtf
        })
    
    return results
```

6. **RTF 평가 기준**
```python
def evaluate_rtf_performance(rtf):
    if rtf <= 0.3:
        return "매우 우수 (초실시간 처리 가능)"
    elif rtf <= 0.7:
        return "우수 (실시간 처리에 여유 있음)"
    elif rtf < 1:
        return "양호 (실시간 처리 가능)"
    else:
        return "부적합 (실시간 처리 불가)"
```

7. **시각화 도구**
```python
def visualize_rtf_performance(rtf_results):
    plt.figure(figsize=(10, 6))
    
    times = [r['timestamp'] for r in rtf_results]
    rtfs = [r['rtf'] for r in rtf_results]
    
    plt.plot(times, rtfs)
    plt.axhline(y=1, color='r', linestyle='--', label='Real-time threshold')
    plt.title('RTF Performance Over Time')
    plt.xlabel('Time')
    plt.ylabel('RTF')
    plt.grid(True)
    plt.legend()
    plt.show()
```

8. **주의사항**
- CPU/GPU 상태 고려
- 시스템 부하 상태 확인
- 메모리 사용량 모니터링
- 배치 크기의 영향

9. **성능 최적화 팁**
- 배치 처리 활용
- GPU 메모리 최적화
- 모델 경량화
- 병렬 처리 구현

10. **실제 적용 사례**
- 실시간 음성 처리
- 스트리밍 서비스
- 실시간 번역
- 음성 인터페이스

RTF는 실시간 처리가 필요한 음성 처리 시스템에서 매우 중요한 평가 지표입니다. RTF < 1 이 실시간 처리의 기본 조건이지만, 안정적인 서비스를 위해서는 더 낮은 RTF 값을 목표로 하는 것이 좋습니다.