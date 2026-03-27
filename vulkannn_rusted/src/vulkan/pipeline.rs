use ash::vk;

pub struct DescriptorSetPool {
    pub sets: Vec<vk::DescriptorSet>,
    pub current: usize,
}

impl DescriptorSetPool {
    pub fn next(&mut self) -> vk::DescriptorSet {
        let set = self.sets[self.current];
        self.current = (self.current + 1) % self.sets.len();
        set
    }
}
