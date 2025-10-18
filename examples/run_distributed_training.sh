#!/bin/bash

# PiKV Distributed Training Scripts
# This script provides easy commands to run different distributed training scenarios

set -e

# Default parameters
NPROC_PER_NODE=4
MASTER_PORT=29500
CHECKPOINT_DIR="./checkpoints"
LOG_DIR="./logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if CUDA is available
check_cuda() {
    if ! python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        print_error "CUDA is not available. Please ensure CUDA is properly installed."
        exit 1
    fi
    print_success "CUDA is available"
}

# Function to check if required packages are installed
check_dependencies() {
    print_info "Checking dependencies..."
    
    # Check PyTorch
    if ! python -c "import torch" 2>/dev/null; then
        print_error "PyTorch is not installed"
        exit 1
    fi
    
    # Check DeepSpeed
    if ! python -c "import deepspeed" 2>/dev/null; then
        print_warning "DeepSpeed is not installed. Install with: pip install deepspeed"
    fi
    
    print_success "Dependencies check completed"
}

# Function to create directories
create_directories() {
    print_info "Creating directories..."
    mkdir -p $CHECKPOINT_DIR
    mkdir -p $LOG_DIR
    print_success "Directories created"
}

# Function to run basic distributed training
run_basic_training() {
    print_info "Running basic distributed training..."
    
    torchrun \
        --nproc_per_node=$NPROC_PER_NODE \
        --master_port=$MASTER_PORT \
        examples/distributed_training_example.py \
        --mode basic \
        --num_epochs 3 \
        --steps_per_epoch 50 \
        --batch_size 4 \
        --seq_len 128 \
        --log_interval 5 \
        --save_interval 25 \
        --checkpoint_dir $CHECKPOINT_DIR/basic \
        --metrics_interval 10 \
        2>&1 | tee $LOG_DIR/basic_training.log
    
    print_success "Basic training completed"
}

# Function to run DeepSpeed training with ZeRO-1
run_deepspeed_zero1() {
    print_info "Running DeepSpeed training with ZeRO-1..."
    
    torchrun \
        --nproc_per_node=$NPROC_PER_NODE \
        --master_port=$MASTER_PORT \
        examples/deepspeed_training_example.py \
        --zero_stage 1 \
        --num_epochs 3 \
        --steps_per_epoch 50 \
        --batch_size 4 \
        --seq_len 128 \
        --log_interval 5 \
        --save_interval 25 \
        --checkpoint_dir $CHECKPOINT_DIR/deepspeed_zero1 \
        --analysis_interval 10 \
        2>&1 | tee $LOG_DIR/deepspeed_zero1.log
    
    print_success "DeepSpeed ZeRO-1 training completed"
}

# Function to run DeepSpeed training with ZeRO-2
run_deepspeed_zero2() {
    print_info "Running DeepSpeed training with ZeRO-2..."
    
    torchrun \
        --nproc_per_node=$NPROC_PER_NODE \
        --master_port=$MASTER_PORT \
        examples/deepspeed_training_example.py \
        --zero_stage 2 \
        --offload_optimizer \
        --num_epochs 3 \
        --steps_per_epoch 50 \
        --batch_size 4 \
        --seq_len 128 \
        --log_interval 5 \
        --save_interval 25 \
        --checkpoint_dir $CHECKPOINT_DIR/deepspeed_zero2 \
        --analysis_interval 10 \
        2>&1 | tee $LOG_DIR/deepspeed_zero2.log
    
    print_success "DeepSpeed ZeRO-2 training completed"
}

# Function to run DeepSpeed training with ZeRO-3
run_deepspeed_zero3() {
    print_info "Running DeepSpeed training with ZeRO-3..."
    
    torchrun \
        --nproc_per_node=$NPROC_PER_NODE \
        --master_port=$MASTER_PORT \
        examples/deepspeed_training_example.py \
        --zero_stage 3 \
        --offload_optimizer \
        --offload_param \
        --num_epochs 3 \
        --steps_per_epoch 50 \
        --batch_size 4 \
        --seq_len 128 \
        --log_interval 5 \
        --save_interval 25 \
        --checkpoint_dir $CHECKPOINT_DIR/deepspeed_zero3 \
        --analysis_interval 10 \
        2>&1 | tee $LOG_DIR/deepspeed_zero3.log
    
    print_success "DeepSpeed ZeRO-3 training completed"
}

# Function to run MoE training with DeepSpeed
run_moe_training() {
    print_info "Running MoE training with DeepSpeed..."
    
    torchrun \
        --nproc_per_node=$NPROC_PER_NODE \
        --master_port=$MASTER_PORT \
        examples/deepspeed_training_example.py \
        --enable_moe \
        --zero_stage 3 \
        --offload_optimizer \
        --offload_param \
        --moe_expert_count 8 \
        --moe_top_k 2 \
        --num_epochs 3 \
        --steps_per_epoch 50 \
        --batch_size 4 \
        --seq_len 128 \
        --log_interval 5 \
        --save_interval 25 \
        --checkpoint_dir $CHECKPOINT_DIR/moe_training \
        --analysis_interval 10 \
        2>&1 | tee $LOG_DIR/moe_training.log
    
    print_success "MoE training completed"
}

# Function to run performance comparison
run_performance_comparison() {
    print_info "Running performance comparison..."
    
    # Run all training modes and compare
    echo "=== Performance Comparison ===" > $LOG_DIR/performance_comparison.log
    
    echo "Basic Training:" >> $LOG_DIR/performance_comparison.log
    run_basic_training >> $LOG_DIR/performance_comparison.log 2>&1
    
    echo "DeepSpeed ZeRO-1:" >> $LOG_DIR/performance_comparison.log
    run_deepspeed_zero1 >> $LOG_DIR/performance_comparison.log 2>&1
    
    echo "DeepSpeed ZeRO-2:" >> $LOG_DIR/performance_comparison.log
    run_deepspeed_zero2 >> $LOG_DIR/performance_comparison.log 2>&1
    
    echo "DeepSpeed ZeRO-3:" >> $LOG_DIR/performance_comparison.log
    run_deepspeed_zero3 >> $LOG_DIR/performance_comparison.log 2>&1
    
    echo "MoE Training:" >> $LOG_DIR/performance_comparison.log
    run_moe_training >> $LOG_DIR/performance_comparison.log 2>&1
    
    print_success "Performance comparison completed. Results in $LOG_DIR/performance_comparison.log"
}

# Function to show help
show_help() {
    echo "PiKV Distributed Training Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  basic              Run basic distributed training"
    echo "  deepspeed-zero1    Run DeepSpeed training with ZeRO-1"
    echo "  deepspeed-zero2    Run DeepSpeed training with ZeRO-2"
    echo "  deepspeed-zero3    Run DeepSpeed training with ZeRO-3"
    echo "  moe                Run MoE training with DeepSpeed"
    echo "  compare            Run performance comparison of all modes"
    echo "  check              Check dependencies and CUDA availability"
    echo "  help               Show this help message"
    echo ""
    echo "Options:"
    echo "  --nproc_per_node N    Number of processes per node (default: 4)"
    echo "  --master_port PORT    Master port for distributed training (default: 29500)"
    echo "  --checkpoint_dir DIR  Checkpoint directory (default: ./checkpoints)"
    echo "  --log_dir DIR         Log directory (default: ./logs)"
    echo ""
    echo "Examples:"
    echo "  $0 basic"
    echo "  $0 deepspeed-zero3"
    echo "  $0 moe"
    echo "  $0 compare"
    echo "  $0 --nproc_per_node 8 deepspeed-zero3"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --nproc_per_node)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --checkpoint_dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --log_dir)
            LOG_DIR="$2"
            shift 2
            ;;
        basic)
            COMMAND="basic"
            shift
            ;;
        deepspeed-zero1)
            COMMAND="deepspeed-zero1"
            shift
            ;;
        deepspeed-zero2)
            COMMAND="deepspeed-zero2"
            shift
            ;;
        deepspeed-zero3)
            COMMAND="deepspeed-zero3"
            shift
            ;;
        moe)
            COMMAND="moe"
            shift
            ;;
        compare)
            COMMAND="compare"
            shift
            ;;
        check)
            COMMAND="check"
            shift
            ;;
        help|--help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_info "PiKV Distributed Training Script"
    print_info "NPROC_PER_NODE: $NPROC_PER_NODE"
    print_info "MASTER_PORT: $MASTER_PORT"
    print_info "CHECKPOINT_DIR: $CHECKPOINT_DIR"
    print_info "LOG_DIR: $LOG_DIR"
    
    # Check dependencies and CUDA
    check_dependencies
    check_cuda
    
    # Create directories
    create_directories
    
    # Execute command
    case $COMMAND in
        basic)
            run_basic_training
            ;;
        deepspeed-zero1)
            run_deepspeed_zero1
            ;;
        deepspeed-zero2)
            run_deepspeed_zero2
            ;;
        deepspeed-zero3)
            run_deepspeed_zero3
            ;;
        moe)
            run_moe_training
            ;;
        compare)
            run_performance_comparison
            ;;
        check)
            print_success "All checks passed"
            ;;
        *)
            print_error "No command specified"
            show_help
            exit 1
            ;;
    esac
    
    print_success "Script completed successfully"
}

# Run main function
main "$@"
