import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

class CifarLabeler:
    def __init__(self, loader):
        self.loader = iter(loader)
        self.correct = 0
        self.total = 0
        self.current_label = None
        self.last_result = "Ready" # Last answer result text

        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        plt.subplots_adjust(bottom=0.3, top=0.85)
        
        # UI text elements
        self.score_text = self.fig.text(0.5, 0.92, "", ha='center', fontsize=12, fontweight='bold', color='blue')
        self.result_text = self.fig.text(0.5, 0.88, "Click a button to start", ha='center', fontsize=10)
        
        self.buttons = []
        self.setup_buttons()
        self.next_image()

    def setup_buttons(self):
        # 2 lines of 5 buttons each
        for i, name in enumerate(class_names):
            col = i % 5
            row = i // 5
            ax_btn = plt.axes([0.05 + col * 0.18, 0.18 - row * 0.08, 0.15, 0.06])
            btn = Button(ax_btn, name, color='lightgray', hovercolor='skyblue')
            btn.on_clicked(lambda event, idx=i: self.check_answer(idx))
            self.buttons.append(btn)

    def update_ui_text(self):
        acc = (self.correct / self.total * 100) if self.total > 0 else 0
        # Update the score text with accuracy, correct count, and total count
        self.score_text.set_text(f"Accuracy: {acc:.1f}%  |  Correct: {self.correct}  |  Total: {self.total}")
        self.result_text.set_text(f"Last Answer: {self.last_result}")

    def next_image(self):
        try:
            image, label = next(self.loader)
            self.current_label = label.item()
            
            self.ax.clear()
            # (C, H, W) -> (H, W, C)
            img_show = image.squeeze().permute(1, 2, 0).numpy()
            self.ax.imshow(img_show)
            self.ax.axis('off')
            
            self.update_ui_text()
            plt.draw()
        except StopIteration:
            self.score_text.set_text("FINISHED! Final Score: " + self.score_text.get_text())
            plt.draw()

    def check_answer(self, user_choice):
        # Check if the user's choice matches the true label
        is_correct = (user_choice == self.current_label)
        true_name = class_names[self.current_label]
        
        if is_correct:
            self.correct += 1
            self.last_result = f"Correct! It was '{true_name}'"
        else:
            self.last_result = f"Wrong! It was '{true_name}'"
        
        self.total += 1
        
        # Feedback every 20 images
        if self.total % 20 == 0:
            acc = (self.correct / self.total) * 100
            print(f"\n[Check-point] Total: {self.total} | Accuracy: {acc:.2f}%")
            
        self.next_image()

if __name__ == "__main__":
    print("--- CIFAR-10 Human Benchmark Tool ---")
    print("Look at the image and click the corresponding category.")
    labeler = CifarLabeler(testloader)
    plt.show()
